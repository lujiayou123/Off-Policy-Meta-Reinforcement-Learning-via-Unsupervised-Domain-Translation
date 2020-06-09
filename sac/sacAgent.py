import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac.utils import soft_update, hard_update
from sac.model import GaussianPolicy, SACQNetwork ,DeterministicPolicy


class SACAgent(object):
    def __init__(self, num_inputs, action_space):#  SACAgent(env.observation_space.shape[0], env.action_space)

        self.gamma = 0.99  # γ
        self.tau = 0.005  # τ
        self.alpha = 1  # α

        self.policy_type = "Gaussian"  # 策略类型，高斯随机策略、确定性策略
        self.target_update_interval = 1  # target network更新间隔
        self.automatic_entropy_tuning = False  # 自动调熵

        self.device = torch.device("cpu")

        self.critic = SACQNetwork(num_inputs=num_inputs, num_actions=action_space.shape[0], hidden_dim=256).to(device=self.device)  # Critic Network，Q网络
        self.critic_optim = Adam(self.critic.parameters(), lr=0.0003)

        self.critic_target = SACQNetwork(num_inputs=num_inputs, num_actions=action_space.shape[0], hidden_dim=256).to(device=self.device)  # Target Q Network
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()  # torch.prod(input) : 返回所有元素的乘积
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)#log alpha=0 ，alpha=1
                self.alpha_optim = Adam([self.log_alpha], lr=0.0003)

            self.policy = GaussianPolicy(num_inputs=num_inputs, num_actions=action_space.shape[0], hidden_dim=256, action_space=action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=0.0003)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs=num_inputs, num_actions=action_space.shape[0], hidden_dim=256, action_space=action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=0.0003)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        # print(state)
        action, _, _ = self.policy.sample(state)  # action, log_prob, torch.tanh(mean)
        # print(action)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)  # 下一个状态下采取的动作，下一个状态的log概率
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)  # Q1目标值和Q2目标值
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (
                min_qf_next_target)  # r(st,at) + γ(𝔼st+1~p[V(st+1)]))

        qf1, qf2 = self.critic(state_batch,action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2],下一状态的值是target network算出来的
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)  # 动作，动作的对数概率

        qf1_pi, qf2_pi = self.critic(state_batch, pi)  # 动作的Q值
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()  # E[-αlogπ(at|st)-αH]

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            print(self.alpha)
            # alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            # alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        # if updates % self.target_update_interval == 0:  # 每隔几步更新target network
        soft_update(self.critic_target, self.critic, self.tau)

        # return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
###############################################################################################################
    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

