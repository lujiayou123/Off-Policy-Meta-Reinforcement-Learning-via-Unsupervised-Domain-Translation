import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu

def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


class PEARLAgent(nn.Module):

    def __init__(self,
                 latent_dim,
                 context_encoder,
                 policy,
                 **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.context_encoder = context_encoder
        self.policy = policy

        self.recurrent = kwargs['recurrent']
        self.use_ib = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))#均值 mu
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))#方差 sigma2

        self.clear_z()

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = ptu.zeros(num_tasks, self.latent_dim)#均值
        if self.use_ib:
            var = ptu.ones(num_tasks, self.latent_dim)#方差
        else:
            var = ptu.zeros(num_tasks, self.latent_dim)
        # print("mu:{},var:{}".format(mu,var))
        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        self.context_encoder.reset(num_tasks)

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        '''
        相当于用context更新一下z的分布,并从更新后的分布中采样z
        context作为encoder的输入,encoder前向传播得到params
        params的前5维作为mu,后5维做softplus作为sigma_squared
        mu和sigma_squared作为_product_of_gaussians函数的参数,得到z_params
        z_params[0]就是均值,z_params[1]就是方差
        '''
    ###############################################
        params = self.context_encoder(context)#[1,256,10][1,batch_size,latent_dim*2]
    ###############################################
        #context size : [1,256,63]
        params = params.view(context.size(0), -1, self.context_encoder.output_size)#params的size标准化
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            # params[1,256,10] mu[1,256,latent_dim]
            mu = params[..., :self.latent_dim]#前5个
            #softplus: log(1+e的x次方)
            sigma_squared = F.softplus(params[..., self.latent_dim:])#后5个
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            #[tensor(1,5),tensor(1,5)],mean,var
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            self.z_means = torch.mean(params, dim=1)
        self.sample_z()

    def sample_z(self):
        if self.use_ib:
            # m表示均值,s表示标准差
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means

    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        obs = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs, z], dim=1)
        #return self.policy.get_action(in_, deterministic=deterministic)
        # action, _, _ =self.policy.sample(in_)
        # return action
        return self.policy.get_action(in_)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def forward(self, obs, context):
        ''' given context, get statistics under the current policy of a set of observations '''
        self.infer_posterior(context)#利用context更新后验分布
        self.sample_z()#从新的分布中采样z

        task_z = self.z

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)

        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1)#in_表示obs与z的拼接
        policy_outputs = self.policy(in_)
        # mean, log_std = self.policy(in_)#forward,action, mean, log_std, log_prob, expected_log_prob, std, mean_action_log_prob, x_t
        # action, log_prob, _ = self.policy.sample()
        # policy_outputs = action,mean,log_std,log_prob
        return policy_outputs, task_z#action, mean, log_std, log_prob, expected_log_prob, std,mean_action_log_prob, pre_tanh_value，任务隐变量Z

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z mean eval'] = z_mean
        eval_statistics['Z variance eval'] = z_sig

    @property
    def networks(self):
        return [self.context_encoder, self.policy]

# if __name__ == '__main__':
#     env = gym.make("Humanoid-v2")
#     obs_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]
#     latent_dim = 5
#     reward_dim = 1
#     net_size =200
#     print("obs_dim:{},action_dim:{},latent_dim:{},reward_dim:{}".format(obs_dim,action_dim,latent_dim,reward_dim))
#     context_encoder = MlpEncoder(  # 上下文编码器
#         hidden_sizes=[200, 200, 200],  # 3个200的隐藏层
#         input_size=2*obs_dim+action_dim+reward_dim,  # 输入层维度为s,a,r,s'维度之和27,13,1,27
#         output_size=latent_dim*2,  # 33行，context维度
#     )
#     policy = GaussianPolicy(obs_dim+latent_dim, action_dim, net_size, env.action_space)
#     agent = PEARLAgent(latent_dim=latent_dim,context_encoder=context_encoder,policy=policy)
#     print("z:{}".format(agent.z))


