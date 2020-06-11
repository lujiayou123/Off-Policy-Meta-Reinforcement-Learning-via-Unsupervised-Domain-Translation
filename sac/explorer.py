import numpy as np
import time
import torch
from rlkit.samplers.util import rollout
from rlkit.torch.sac.policies import MakeDeterministic
from sac.sacAgent import SACAgent

def rollout(env, agent, max_path_length, random_steps=0):
    observations = []
    actions = []
    rewards = []
    terminals = []
    next_observations = []
    o = env.reset()
    next_o = None
    path_length = 0

    while path_length < max_path_length:

        if path_length < random_steps:
            print("random")
            a = env.action_space.sample()
        else:
            a = agent.select_action(o)

        next_o, r, d, _ = env.step(a)
        observations.append(o)
        actions.append(a)
        rewards.append(r)
        terminals.append(d)
        next_observations.append(next_o)


        path_length += 1
        if d:
            break
        o = next_o

    # actions = np.array(actions)
    # time.sleep(5)
    # if len(actions.shape) == 1:
    #     actions = np.expand_dims(actions, 1)
    # observations = np.array(observations)
    # if len(observations.shape) == 1:
    #     observations = np.expand_dims(observations, 1)
    #     next_o = np.array([next_o])
    # next_observations = np.vstack(
    #     (
    #         observations[1:, :],
    #         np.expand_dims(next_o, 0)
    #     )
    # )
    return dict(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=np.array(next_observations),
        terminals=np.array(terminals).reshape(-1, 1),
    )

class SACExplorer(object):
    """
    A Explorer designed for exploring , focus on collecting informative experiences.
    It's based on sac policy.

    Explorer aims at answering the following question:
    How to collect the minimum numbers of experiences that is enough for task inference ?
    1.reward1 is designed to encourage exploration
    2.reward2 is designed to ensure quality , since area with higher rewards consists of more task information,
which is especially important in sparse reward environments, eg. maze.
########################################################################################################################
Algorithm:
    explorer与环境交互，收集经验给encoder更新task belief，重复K轮，encoder给出最终的task belief给actor
    Loop K次:
        explorer与环境交互，收集n条经验(minimum but informative)
        把收集到的n条经验喂给encoder
        encoder输出task belief z
    把K次确定后的z给actor做task control
    """
    def __init__(self, env, max_path_length, latent_dim):
        self.env = env
        '''
        把z接到observation上面,作为输入
        '''
        self.agent = SACAgent(env.observation_space.shape[0] + latent_dim, env.action_space)
        self.max_path_length = max_path_length
        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))  # 均值 mu
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))  # 方差 sigma2

    def obtain_samples(self, max_samples, max_trajs, random_steps):
        paths = []
        n_steps_total = 0
        n_trajs = 0
        while n_steps_total < max_samples and n_trajs < max_trajs:
            path = rollout(env=self.env,agent=self.agent,max_path_length=self.max_path_length,random_steps=random_steps)
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1
        return paths, n_steps_total, n_trajs

if __name__ == '__main__':
    pass