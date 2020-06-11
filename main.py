import argparse
import datetime
import gym
import numpy as np
import torch
print("torch version:",torch.__version__)
from sac.sacAgent import SACAgent
from sac.replay_memory import ReplayMemory
from sac.explorer import SACExplorer

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env-name', default="Humanoid-v2",
                    help='name of the environment to run')
parser.add_argument('--policy', default="Gaussian",
                    help='algorithm to use: Gaussian | Deterministic')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy a policy every 10 episode (default:True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Temperature parameter α automaically adjusted.')
parser.add_argument('--seed', type=int, default=456, metavar='N',
                    help='random seed (default: 456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=500, metavar='N',#10000
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', type=bool, default=True,
                    help='run on CUDA (default: False)')
parser.add_argument('--cluster', type=bool, default=False,
                    help='cluster reward')
parser.add_argument('--max_path_length', type=int, default=200,
                    help='if steps > max_path_length,episode ends')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# env.seed(args.seed)

# Agent
agent = SACAgent(env.observation_space.shape[0], env.action_space)

# Memory
exp_memory = ReplayMemory(args.replay_size)
# Training Loop
total_numsteps = 0
total_episodes = 0
updates = 0

# Explorer Test
# explorer = SACExplorer(env=env,max_path_length=args.max_path_length)
# episodes, n_steps_total, n_trajs = explorer.obtain_samples(max_samples=2000,max_trajs=20,random_steps=args.start_steps)
# print("total steps:",n_steps_total)
# print("total episodes:",n_trajs)
# print(episodes[4]["observations"].shape)#(42, 376),这条trajectory总共走了42steps,observation维度为376
# print(episodes[4]["actions"].shape)
# print(episodes[4]["rewards"].shape)
# print(episodes[4]["terminals"].shape)
# print(episodes[4]["next_observations"].shape)
# for i in range(n_trajs):
#     print(np.sum(episodes[i]["rewards"]))


for i_episode in range(5):#itertools.count(1)创建一个从start开始每次的步长是step的无穷序列
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        #先进行start_steps步随机探索
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy
        # 当前经验池大小超过batch_size就可以开始更新
        # action = agent.select_action(state)  # Sample action from policy
        if len(exp_memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                agent.update_parameters(memory=exp_memory, batch_size=256, updates=updates)
                updates += 1

        next_state, reward, done, _ = env.step(action) # Step，走一步，获得下一个状态值，奖赏，是否终止
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        exp_memory.push(state, action, reward, next_state, mask) # Append transition to memory
        state = next_state
    # if total_numsteps > args.num_steps:#终止条件
    if total_numsteps > 1000000:  # 终止条件
        break
    if args.start_steps > total_numsteps:
        print("random sample")
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {},average steps:{}".format(i_episode+1, total_numsteps, episode_steps, round(episode_reward, 2),total_numsteps/(i_episode+1)))
    #Test
    # if i_episode % 10 == 0 and args.eval == True:
    #     avg_reward = 0.
    #     episodes = 10
    #     for _  in range(episodes):
    #         state = env.reset()
    #         episode_reward = 0
    #         done = False
    #         while not done:
    #             action = agent.select_action(state, eval=True)
    #
    #             next_state, reward, done, _ = env.step(action)
    #             episode_reward += reward
    #
    #
    #             state = next_state
    #         avg_reward += episode_reward
    #     avg_reward /= episodes
    #
    #     print("----------------------------------------")
    #     print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
    #     print("----------------------------------------")

env.close()
# print(len(exp_memory))
# for i in range(5):
#     print(i)
#     print(exp_memory.buffer[0][i])#第一条经验的s,a,r,s',d

