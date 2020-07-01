from collections import OrderedDict
import numpy as np
import time
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm

import torch.nn.functional as F
from torch.optim import Adam

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class PEARLSoftActorCritic(MetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,

            policy_lr=3e-4,
            alpha = 0.15,
            explorer_alpha = 1,
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
            explorer = nets[3],
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
        self.explorer_alpha = explorer_alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.recurrent = recurrent
        self.latent_dim = latent_dim

        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards

        # for rl agent
        self.critic, self.critic_target = nets[1:3]#q1,q2,target q1,target q2
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        hard_update(self.critic_target,self.critic)
        self.policy_optimizer = Adam(self.agent.policy.parameters(), lr=policy_lr)#都是高斯策略

        self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape)).to("cuda").item()  # torch.prod(input) : 返回所有元素的乘积
        self.log_alpha = torch.zeros(1, requires_grad=True, device="cuda")
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

        #for explorer
        self.exploration_critic, self.exploration_critic_target = nets[4:6]
        self.exploration_critic_optimizer = Adam(self.exploration_critic.parameters(), lr=lr)
        hard_update(self.exploration_critic_target, self.exploration_critic)
        self.exploration_policy_optimizer = Adam(self.explorer.policy.parameters(), lr=policy_lr)  # 都是高斯策略

        #for both
        self.debug = False
        self.original_plan = True

        # self.context_optimizer = optimizer_class(
        #     self.agent.context_encoder.parameters(),
        #     lr=context_lr,
        # )
        self.context_optimizer = optimizer_class(
            self.explorer.context_encoder.parameters(),
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
    def sample_data(self, indices, explorer=False):
        ''' sample data from replay buffers to construct a training meta-batch '''
        # collect data from multiple tasks for the meta-batch
        obs, actions, rewards, next_obs, terms = [], [], [], [], []
        for idx in indices:
            if explorer:
                batch = ptu.np_to_pytorch_batch(self.exploration_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent))
            else:
                batch = ptu.np_to_pytorch_batch(self.rl_replay_buffer.random_batch(idx, batch_size=self.batch_size))
            o = batch['observations'][None, ...]
            a = batch['actions'][None, ...]
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
        if self.debug:
            print("obs_shape:",obs.shape)
        return [obs, actions, rewards, next_obs, terms]



    ##### Training #####
    def _do_training(self, indices):#对indices中的任务进行训练

        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size#其实就是一次
        if self.debug:
            print("num_updates:", num_updates)
        explorer_batch = self.sample_data(indices, explorer=True)#用explorer收集的数据训练encoder,以及explorer本身

        # zero out context and hidden encoder state
        #用采集到的数据更新参数,
        if self.original_plan:
            self.agent.clear_z(num_tasks=len(indices))
            self.explorer.clear_z(num_tasks=len(indices))

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            mini_batch = [x[:, i * mb_size: i * mb_size + mb_size, :] for x in explorer_batch]
            obs_explore, act_explore, rewards_explore, next_obs_explore, _ = mini_batch
            context = self.prepare_encoder_data(obs_explore, act_explore, rewards_explore,next_obs_explore)#concat,作为输入
            self.explorer_take_step(indices,context)
            self.rl_take_step(indices, context)

            # stop backprop
            self.agent.detach_z()

    def explorer_take_step(self, indices, context):
        num_tasks = len(indices)
        #sample data from exploration buffer for explorer training
        obs, actions, rewards, next_obs, terms = self.sample_data(indices=indices, explorer=True)#从Replay Buffer采集数据，s,a,r,s',d

        # run inference in networks
        if self.original_plan:
            #context作为输入encoder前向传播得到分布的均值方差,从中采样的z和obs concat作为policy的输入前向传播得到policy_outputs
            #Q:explorer需要用obs concat z吗?
            policy_outputs, task_z = self.explorer(obs, context)
        else:
            print(self.explorer.z)
            print(self.agent.z)
            task_z = self.agent.z
            policy_outputs, _ = self.agent(obs, context)

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]  # 下一个状态下策略所采取的动作，其log概率 line 63

        # flattens out the task dimension:
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q networks
        # encoder will only get gradients from Q nets
        q1, q2 = self.exploration_critic(obs, actions, task_z)  # forward

        # get targets for q network
        with torch.no_grad():
            q1_next_target, q2_next_target = self.exploration_critic_target(next_obs, new_actions, task_z)  # target q               line 64
            min_qf_next_target = torch.min(q1_next_target, q2_next_target)  # 计算较小的target Q         line 65

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.explorer.compute_kl_div()  # context_encoder前向传播得到μ和σ，计算该分布与先验分布的kl散度
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # qf and encoder update (note that encoder does not get grads from policy)
        self.exploration_critic_optimizer.zero_grad()

        rewards_flat = rewards.view(self.embedding_batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        # q=r+(1-d)γ(Vst+1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * (min_qf_next_target - self.explorer_alpha * log_pi)
        q1_loss = torch.mean((q1 - q_target) ** 2)
        q2_loss = torch.mean((q2 - q_target) ** 2)
        qf_loss = q1_loss + q2_loss
        qf_loss.backward(retain_graph=True)
        self.exploration_critic_optimizer.step()
        self.context_optimizer.step()

        q1_pi, q2_pi = self.exploration_critic(obs, new_actions, task_z)  # 动作的Q值                                                line 74
        min_q_pi = torch.min(q1_pi, q2_pi)  # line 75

        #update target Q network
        soft_update(self.exploration_critic_target, self.exploration_critic, self.soft_target_tau)

        policy_loss = ((self.explorer_alpha * log_pi) - min_q_pi).mean()
        # mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        # std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        # pre_tanh_value = policy_outputs[-1]
        # pre_activation_reg_loss = self.policy_pre_activation_weight * (
        #     (pre_tanh_value ** 2).sum(dim=1).mean()
        # )
        # policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        # policy_loss = policy_loss + policy_reg_loss

        self.exploration_policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.exploration_policy_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.explorer.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.explorer.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['explorer Q Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['explorer Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'Explorer Policy mu',
            #     ptu.get_numpy(policy_mean),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'Explorer Policy log std',
            #     ptu.get_numpy(policy_log_std),
            # ))

    def rl_take_step(self, indices, context):
        num_tasks = len(indices)
        #sample data from rl buffer for rl training
        obs, actions, rewards, next_obs, terms = self.sample_data(indices)#从Replay Buffer采集数据，s,a,r,s',d

        # run inference in networks
        if self.original_plan:#encoder前向传播得到分布的均值方差,采样的z和obs concat作为policy的输入前向传播得到policy_outputs
            #Q:更新参数时需要采样z吗?
            policy_outputs, task_z = self.agent(obs, context)
        else:
            print(self.explorer.z)
            print(self.agent.z)
            task_z = self.agent.z
            policy_outputs, _ = self.agent(obs, context)

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
            q1_next_target, q2_next_target = self.critic_target(next_obs, new_actions,task_z)  # target q        line 64
            min_qf_next_target = torch.min(q1_next_target, q2_next_target)             # 计算较小的target Q        line 65

#在explorer时已经算过了encoder,rl agent还需要吗?

        # KL constraint on z if probabilistic
        # self.context_optimizer.zero_grad()
        # if self.use_information_bottleneck:
        #     kl_div = self.agent.compute_kl_div()  # context_encoder前向传播得到μ和σ，计算该分布与先验分布的kl散度
        #     kl_loss = self.kl_lambda * kl_div
        #     kl_loss.backward(retain_graph=True)

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

        # self.context_optimizer.step()

        q1_pi, q2_pi = self.critic(obs, new_actions, task_z)  # 动作的Q值                                                line 74
        min_q_pi = torch.min(q1_pi, q2_pi)  # line 75

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
            #log_alpha是cuda,log_pi不是,target_entropy是float
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()  # E[-αlogπ(at|st)-αH]
            self.alpha_optim.zero_grad()
            alpha_loss.backward(retain_graph=True)
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        self.eval_statistics['RL Q Loss'] = np.mean(ptu.get_numpy(qf_loss))
        self.eval_statistics['RL Policy Loss'] = np.mean(ptu.get_numpy(
            policy_loss
        ))
        if self.automatic_entropy_tuning:
            self.eval_statistics['Alpha Loss'] = np.mean(ptu.get_numpy(alpha_loss))
            self.eval_statistics['Alpha'] = np.mean(ptu.get_numpy(self.alpha))
        self.eval_statistics.update(create_stats_ordered_dict(
            'RL Q1 Predictions',
            ptu.get_numpy(q1),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'RL Q2 Predictions',
            ptu.get_numpy(q2),
        ))
        # self.eval_statistics.update(create_stats_ordered_dict(
        #     'RL Log Pis',
        #     ptu.get_numpy(log_pi),
        # ))
        # self.eval_statistics.update(create_stats_ordered_dict(
        #     'RL Policy mu',
        #     ptu.get_numpy(policy_mean),
        # ))
        # self.eval_statistics.update(create_stats_ordered_dict(
        #     'RL Policy log std',
        #     ptu.get_numpy(policy_log_std),
        # ))
        # # save some statistics for eval
        # if self.eval_statistics is None:
        #     # eval should set this to None.
        #     # this way, these statistics are only computed for one batch.
        #     self.eval_statistics = OrderedDict()


    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            critic = self.critic.state_dict(),
            critic_target = self.critic_target.state_dict(),
            policy=self.agent.policy.state_dict(),
            context_encoder=self.explorer.context_encoder.state_dict(),
            explorer_critic = self.exploration_critic.state_dict(),
            explorer_critic_target = self.exploration_critic_target.state_dict(),
            explorer_policy = self.explorer.policy.state_dict()
        )
        return snapshot
