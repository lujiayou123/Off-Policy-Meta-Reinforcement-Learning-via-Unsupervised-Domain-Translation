from collections import OrderedDict
import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm

import torch.nn.functional as F
from torch.optim import Adam
from sac.utils import soft_update,hard_update


class PEARLSoftActorCritic(MetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,

            policy_lr=3e-4,
            qf_lr=1e-3,
            vf_lr=1e-3,
            alpha = 0.15,
            automatic_entropy_tuning = True,
            lr=3e-4,
            context_lr=3e-4,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            sparse_rewards=False,

            #soft_target_tau=1e-2,
            soft_target_tau=0.005,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )

        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.recurrent = recurrent
        self.latent_dim = latent_dim
        # self.qf_criterion = nn.MSELoss()
        # self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards

        # self.qf1, self.qf2, self.vf = nets[1:]
        self.critic,self.critic_target = nets[1:]#q1,q2,target q1,target q2
        # self.target_vf = self.vf.copy()
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        hard_update(self.critic_target,self.critic)
        self.policy_optimizer = Adam(self.agent.policy.parameters(), lr=policy_lr)


        self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape)).to("cuda").item()  # torch.prod(input) : 返回所有元素的乘积
        self.log_alpha = torch.zeros(1, requires_grad=True, device="cuda")
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

        # self.policy_optimizer = optimizer_class(
        #     self.agent.policy.parameters(),
        #     lr=policy_lr,
        # )
        # self.qf1_optimizer = optimizer_class(
        #     self.qf1.parameters(),
        #     lr=qf_lr,
        # )
        # self.qf2_optimizer = optimizer_class(
        #     self.qf2.parameters(),
        #     lr=qf_lr,
        # )
        # self.vf_optimizer = optimizer_class(
        #     self.vf.parameters(),
        #     lr=vf_lr,
        # )
        self.context_optimizer = optimizer_class(
            self.agent.context_encoder.parameters(),
            lr=context_lr,
        )

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.agent] + [self.critic, self.critic_target]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def sample_data(self, indices, encoder=False):
        ''' sample data from replay buffers to construct a training meta-batch '''
        # collect data from multiple tasks for the meta-batch
        obs, actions, rewards, next_obs, terms = [], [], [], [], []
        for idx in indices:
            if encoder:
                batch = ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent))
            else:
                batch = ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size))
            o = batch['observations'][None, ...]
            a = batch['actions'][None, ...]
            if encoder and self.sparse_rewards:
                # in sparse reward settings, only the encoder is trained with sparse reward
                r = batch['sparse_rewards'][None, ...]
            else:
                r = batch['rewards'][None, ...]
            no = batch['next_observations'][None, ...]
            t = batch['terminals'][None, ...]
            obs.append(o)
            actions.append(a)
            rewards.append(r)
            next_obs.append(no)
            terms.append(t)
        obs = torch.cat(obs, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0)
        next_obs = torch.cat(next_obs, dim=0)
        terms = torch.cat(terms, dim=0)
        return [obs, actions, rewards, next_obs, terms]



    ##### Training #####
    def _do_training(self, indices):#对indices中的任务进行训练
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size#其实就是一次

        batch = self.sample_data(indices, encoder=True)#sample训练encoder的数据

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        for i in range(num_updates):
            mini_batch = [x[:, i * mb_size: i * mb_size + mb_size, :] for x in batch]
            obs_enc, act_enc, rewards_enc, _, _ = mini_batch
            context = self.prepare_encoder_data(obs_enc, act_enc, rewards_enc)
            self._take_step(indices, context)

            # stop backprop
            self.agent.detach_z()

    def _take_step(self, indices, context):

        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_data(indices)#从Replay Buffer采集数据，s,a,r,s',d
        # obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # run inference in networks
        policy_outputs, task_z = self.agent(obs, context)  # 策略forward的输出，以及任务隐变量Z
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]  # 下一个状态下策略所采取的动作，其log概率 line 63

        # flattens out the task dimension:
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q networks
        # encoder will only get gradients from Q nets
        q1, q2 = self.critic(obs, actions, task_z)  # forward

        # get targets for use in V and Q updates
        with torch.no_grad():
            q1_next_target, q2_next_target = self.critic_target(next_obs, new_actions,task_z)  # target q               line 64
            min_qf_next_target = torch.min(q1_next_target, q2_next_target)  # 计算较小的target Q         line 65

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()  # context_encoder前向传播得到μ和σ，计算该分布与先验分布的kl散度
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.critic_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * (min_qf_next_target - self.alpha * log_pi)  # q=r+(1-d)γ(Vst+1)
        q1_loss = torch.mean((q1 - q_target) ** 2)
        q2_loss = torch.mean((q2 - q_target) ** 2)
        qf_loss = q1_loss + q2_loss
        qf_loss.backward(retain_graph=True)
        self.critic_optimizer.step()
        self.context_optimizer.step()


        # q1_loss = F.mse_loss(q1,q_target)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]       line 69
        # q2_loss = F.mse_loss(q2,q_target)  # line 70
        # pi, log_pi, _ = self.agent.policy.sample(obs)  # 动作，动作的对数概率
        # print(obs.size())  #[1024,27]
        # print(task_z.size())

        # compute min Q on the new actions
        # in_policy = torch.cat([obs, task_z], 1)
        # pi, _, _, log_pi, _, _, _, _, = self.agent.policy(in_policy)  # line 72
        q1_pi, q2_pi = self.critic(obs, new_actions,task_z)  # 动作的Q值                                                line 74
        min_q_pi = torch.min(q1_pi, q2_pi)  # line 75

        # self.critic_optimizer.zero_grad()
        # q1_loss.backward(retain_graph=True)
        # self.critic_optimizer.step()

        # self.critic_optimizer.zero_grad()
        # q2_loss.backward(retain_graph=True)
        # self.critic_optimizer.step()

        #update target Q network
        soft_update(self.critic_target, self.critic, self.soft_target_tau)

        policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()  # line 77
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()  # E[-αlogπ(at|st)-αH]
            self.alpha_optim.zero_grad()
            alpha_loss.backward(retain_graph=True)
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['Q1 Loss'] = np.mean(ptu.get_numpy(q1_loss))
            self.eval_statistics['Q2 Loss'] = np.mean(ptu.get_numpy(q2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            if self.automatic_entropy_tuning:
                self.eval_statistics['Alpha Loss'] = np.mean(ptu.get_numpy(alpha_loss))
                self.eval_statistics['Alpha'] = np.mean(ptu.get_numpy(self.alpha))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            # qf1=self.qf1.state_dict(),
            # qf2=self.qf2.state_dict(),
            critic = self.critic.state_dict(),
            critic_target = self.critic_target.state_dict(),
            policy=self.agent.policy.state_dict(),
            # vf=self.vf.state_dict(),
            # target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
        )
        return snapshot
