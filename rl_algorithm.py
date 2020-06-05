import abc
from collections import OrderedDict
import time
import numpy as np

from rlkit.core import logger, eval_util
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.in_place import InPlacePathSampler
from rlkit.torch import pytorch_util as ptu
from sac.explorer import SACExplorer


class MetaRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            agent,
            train_tasks,
            eval_tasks,
            meta_batch=64,
            num_iterations=100,
            num_train_steps_per_itr=1000,
            num_initial_steps=100,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_posterior=100,
            num_extra_rl_steps_posterior=100,
            num_evals=10,
            num_steps_per_eval=1000,
            batch_size=1024,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=1000000,
            reward_scale=1,
            num_exp_traj_eval=1,
            update_post_train=1,
            eval_deterministic=True,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False,
            render_eval_paths=False,
            dump_eval_paths=False,
            plotter=None,
    ):
        """
        :param env: training env
        :param agent: agent that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval

        see default experiment config file for descriptions of the rest of the arguments
        """
        self.env = env
        self.agent = agent
        self.exploration_agent = agent # Can potentially use a different policy purely for exploration rather than also solving tasks, currently not being used
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.meta_batch = meta_batch
        self.num_iterations = num_iterations
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.num_initial_steps = num_initial_steps
        self.num_tasks_sample = num_tasks_sample
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.num_evals = num_evals
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.update_post_train = update_post_train
        self.num_exp_traj_eval = num_exp_traj_eval
        self.eval_deterministic = eval_deterministic
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment

        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.dump_eval_paths = dump_eval_paths
        self.plotter = plotter
        # self.alpha = alpha

        self.explorer = Explorer(env=env,max_path_length=self.max_path_length)

        self.sampler = InPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=self.max_path_length,
        )

        # separate replay buffers for
        # - training RL update
        # - training encoder update
        self.replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
            )

        self.enc_replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
        )

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        # self._current_path_builder = PathBuilder()
        self._exploration_paths = []

    def train(self):
        '''
        meta-training loop
        '''
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)#像是保存参数到文件
        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for iteration in range(self.num_iterations):
            print("\nIteration:{}".format(iteration+1))
            if iteration == 0:#                                                                算法第一步，初始化每个任务的buffer
                print('\nCollecting initial pool of data for train and eval')
                # temp for evaluating
                for idx in self.train_tasks:#在训练开始之前，为每个任务采集2000条transition
                    self.task_idx = idx#更改当前任务idx
                    self.env.reset_task(idx)#重置任务
                    self.collect_data(self.num_initial_steps, 1, np.inf)#采集num_initial_steps条轨迹c并利用q(z|c)更新self.z
                    # print("task id:", self.task_idx, " env:", self.replay_buffer.env)
                    # print("buffer ", self.task_idx, ":", self.replay_buffer.task_buffers[self.task_idx].__dict__.items())
            # Sample data from train tasks.
            print("\nFinishing collecting initial pool of data")
            print("\nSampling data from train tasks for Meta-training")
            for i in range(self.num_tasks_sample):#对于所有的train_tasks，随机从中取5个，然后为每个任务的buffer采集num_steps_prior + num_extra_rl_steps_posterior条transition
                print("\nSample data , round{}".format(i+1))#为每个任务的enc_buffer采集num_steps_prior条transition
                idx = np.random.randint(len(self.train_tasks))#train_tasks里面随便选一个task
                self.task_idx = idx
                self.env.reset_task(idx)#task重置
                self.enc_replay_buffer.task_buffers[idx].clear()#清除对应的enc_bufffer

                # collect some trajectories with z ~ prior
                if self.num_steps_prior > 0:
                    print("\ncollect some trajectories with z ~ prior")
                    self.collect_data(self.num_steps_prior, 1, np.inf)#利用z的先验采集num_steps_prior条transition
                # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    print(  "\ncollect some trajectories with z ~ posterior")
                    self.collect_data(self.num_steps_posterior, 1, self.update_post_train)#利用后验的z收集轨迹
                # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    print("\ncollect some trajectories for policy update only")
                    self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train, add_to_enc_buffer=False)#利用后验的z收集num_extra_rl_steps_posterior条轨迹，仅用于策略
            print("\nFinishing sample data from train tasks")
            # Sample train tasks and compute gradient updates on parameters.
            print("\nStrating Meta-training ， Episode {}".format(it_))
            for train_step in range(self.num_train_steps_per_itr):#每轮迭代计算num_train_steps_per_itr次梯度              500x2000=1000000
                indices = np.random.choice(self.train_tasks, self.meta_batch)#train_tasks中随机取meta_batch个task , sample RL batch b~B
                if ((train_step + 1) % 500 == 0):
                    print("\nTraining step {}".format(train_step + 1))
                    print("Indices: {}".format(indices))
                    print("alpha:{}".format(self.alpha))
                self._do_training(indices)#梯度下降
                self._n_train_steps_total += 1

            # eval
            self._try_to_eval(iteration)#训练完了，评估模型


    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        pass