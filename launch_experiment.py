"""
Launcher for experiments with PEARL

"""
import os
import pathlib
import numpy as np
import click
import json
import torch

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder ,RNN
from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config


from sac.model import QNetwork,GaussianPolicy,DeterministicPolicy




def experiment(variant):

    # create multi-task environment and sample tasks
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))#创建环境
    tasks = env.get_all_task_idx()#采集任务
    obs_dim = int(np.prod(env.observation_space.shape))#观察空间维度
    action_dim = int(np.prod(env.action_space.shape))#动作空间维度

    # instantiate networks
    latent_dim = variant['latent_size']#隐变量维度
    context_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim#用了信息瓶颈就x2，不用的话latent context就是确定的
    reward_dim = 1#1维奖赏
    net_size = variant['net_size']#网络大小
    recurrent = variant['algo_params']['recurrent']#是否RNN

    encoder_model = RNN if recurrent else MlpEncoder
    #RNN
    # context_encoder = encoder_model(
    #     input_size=obs_dim + action_dim + reward_dim,
    #     hidden_size=200,
    #     num_layers=3,
    #     output_size=context_dim
    # )
    #encoder_model = RecurrentEncoder if recurrent else MlpEncoder#RNN encoder或者MLP encoder（permutation invariant）
    #permutation invariant
    context_encoder = encoder_model(#上下文编码器
        hidden_sizes=[200, 200, 200],#3个200的隐藏层
        input_size=obs_dim + action_dim + reward_dim,#输入层维度为s,a,r维度之和
        output_size=context_dim,#33行，context维度
    )





    # qf1 = FlattenMlp(#多输入按列拍平，Q1网络
    #     hidden_sizes=[net_size, net_size, net_size],
    #     input_size=obs_dim + action_dim + latent_dim,
    #     output_size=1,
    # )
    # qf2 = FlattenMlp(#Q2网络
    #     hidden_sizes=[net_size, net_size, net_size],
    #     input_size=obs_dim + action_dim + latent_dim,
    #     output_size=1,
    # )
    critic = QNetwork(obs_dim + latent_dim, action_dim, net_size)
    # qf_optim = Adam(qf.parameters(),lr=variant['algo_params']['qf_lr'])
    critic_target = QNetwork(obs_dim + latent_dim, action_dim, net_size)
    # hard_update(vf,qf)
    # vf = FlattenMlp(#V值网络
    #     hidden_sizes=[net_size, net_size, net_size],
    #     input_size=obs_dim + latent_dim,
    #     output_size=1,
    # )
    #原本输出action, mean, log_std, log_prob, expected_log_prob, std, mean_action_log_prob, pre_tanh_value,
    policy = GaussianPolicy(obs_dim + latent_dim, action_dim, net_size, env.action_space)#可以输出mean, log_std, action, log_prob, torch.tanh(mean)
    # policy = TanhGaussianPolicy(#策略网络
    #     hidden_sizes=[net_size, net_size, net_size],
    #     obs_dim=obs_dim + latent_dim,
    #     latent_dim=latent_dim,
    #     action_dim=action_dim,
    # )

    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )
    algorithm = PEARLSoftActorCritic(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),#从前往后数，前n_train_tasks个任务
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),#从后往前数，第n_eval_tasks个任务
        # nets=[agent, qf1, qf2, vf],
        nets=[agent,critic,critic_target],
        latent_dim=latent_dim,
        **variant['algo_params']
    )

    # optionally load pre-trained weights
    # if variant['path_to_weights'] is not None:
    #     path = variant['path_to_weights']
    #     context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
    #     qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
    #     qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
    #     vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth')))
    #     # TODO hacky, revisit after model refactor
    #     algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
    #     policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))

    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'])

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    print("State Dim:", obs_dim)
    print("Action Dim:", action_dim)
    print("alpha:",algorithm.alpha)
    print("automatic_entropy_tuning:",algorithm.automatic_entropy_tuning)
    print("\ntasks:{}".format(tasks))
    print("train tasks:{}".format(list(tasks[:variant['n_train_tasks']])))
    print("test tasks:{}".format(list(tasks[-variant['n_eval_tasks']:])))
    algorithm.train()

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.argument('config', default="./configs/humanoid-dir.json")
@click.option('--gpu', default=0)
@click.option('--docker', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
def main(config, gpu, docker, debug):
#读入json参数，有新的就替换旧的
    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu

    experiment(variant)

if __name__ == "__main__":
    main()

