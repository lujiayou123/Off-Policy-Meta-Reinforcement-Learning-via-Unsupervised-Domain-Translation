import abc
from collections import OrderedDict
import time
import numpy as np
import torch
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
            explorer,
            train_tasks,
            eval_tasks,
            meta_batch=64,
            num_iterations=100,
            num_train_steps_per_itr=1000,
            num_initial_steps=100,
            num_exploration_steps=200,
            num_rl_training_steps=400,
            num_random_steps=50,
            num_exploration_episodes=3,
            num_task_inference=5,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_posterior=100,
            num_extra_rl_steps_posterior=100,
            num_evals=10,
            num_steps_per_eval=1000,
            batch_size=1024,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            max_path_length=200,
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
        self.explorer = explorer
        # self.exploration_agent = agent # Can potentially use a different policy purely for exploration rather than also solving tasks, currently not being used
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.meta_batch = meta_batch
        self.num_iterations = num_iterations
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.num_initial_steps = num_initial_steps
        self.num_exploration_steps = num_exploration_steps
        self.num_rl_training_steps = num_rl_training_steps
        self.num_random_steps = num_random_steps
        self.num_exploration_episodes = num_exploration_episodes
        self.num_task_inference = num_task_inference
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

        self.exploration_sampler = SACExplorer(env=env,
                                              agent=explorer,#explorer is an instance of PEARLAgent
                                              max_path_length=self.max_path_length)
        self.sampler = SACExplorer(env=env,
                                    agent=explorer,  # explorer is an instance of PEARLAgent
                                    max_path_length=self.max_path_length)

        # self.sampler = InPlacePathSampler(
        #     env=env,
        #     policy=agent,
        #     max_path_length=self.max_path_length,
        # )

        # separate replay buffers for
        # - training encoder update(collected by explorer SAC)
        # - training RL update(collected by SAC with z)
        self.exploration_replay_buffer = MultiTaskReplayBuffer(
            self.replay_buffer_size,
            env,
            self.train_tasks,
        )
        self.rl_replay_buffer = MultiTaskReplayBuffer(
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

        self.debug = False

    def train(self):
        '''
        meta-training loop

        task exploration以及task inference过程的data都是由explorer提供
        task control采用的是actor
        本质上是两个SAC

        for i in range(K):
            explorer采集n步,存入explorer buffer
            encoder利用explorer buffer里面的data进行task inference,得到task belief z
        清空explorer buffer
        K轮迭代后最终确定的z交给actor

        '''
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)#像是保存参数到文件
        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for iteration in range(self.num_iterations):
            print("\nIteration:{}".format(iteration+1))
#是否需要随机2000步???
###########################################################################################
            if iteration == 0:#                                                                算法第一步，初始化每个任务的buffer
                print('\nCollecting random data for rl training')
                # 为每个任务用explorer采集一下数据,作为第一次task inference的基础
                for idx in self.train_tasks:
                    self.task_idx = idx#更改当前任务idx
                    self.env.reset_task(idx)#重置任务
                    #采集xxx步随机数据用于task inference
                    self.collect_data(num_samples=self.embedding_batch_size,
                                      exploration=True,
                                      random_steps=self.embedding_batch_size)#随机embedding_batch_size防止buffer不够
                    #采集xxx步随机数据用于rl training
                    self.collect_data(num_samples=self.num_initial_steps,
                                      exploration=False,
                                      random_steps=self.num_initial_steps)  #随机xxx步给rl buffer
            print("\nFinishing collecting random steps of data for each task")
############################################################################################
            print("\nSampling data from train tasks for Meta-training")
#刚才各随机采集了1000步
# 利用explorer再采集num_steps_prior步,利用这些data进行task inference
# 利用actor采集num_extra_rl_steps_posterior步,利用这些data以及inference得到的task belief进行task control
            for i in range(self.num_tasks_sample):#对于所有的train_tasks，随机从中取5个，然后为每个任务的buffer采集num_steps_prior + num_extra_rl_steps_posterior条transition
                print("\nData Sampling Round {}/{}:".format(i+1, self.num_tasks_sample))
                idx = np.random.randint(len(self.train_tasks))#train_tasks里面随便选一个task
                print("\nSample data for task {}".format(idx))
                self.task_idx = idx
                self.env.reset_task(idx)  # task重置
                print("\ncollect some trajectories for task inference")
    #每次buffer是否需要重置?
                #encoder is trained only on samples from the prior
                for inference in range(self.num_task_inference):#每200步,infer一次后验,做5轮
                # collect data with explorer for task inference
                # collect data本质上是rollout policy
                    print("\nInference {} :".format(inference+1))
                    #collect experiences with prior
                    self.collect_data(num_samples=self.num_exploration_steps, exploration=True, random_steps=0)#利用z的先验采集experience
                    '''
                    update posterior distribution with collected experiences
                        原先是只用prior采集数据到encoder buffer,采集期间不更新posterior.
                    而在rl data collection时,从posterior中采样,利用采样出来的z来rollout,
                    每采集一个trajectory就更新一次posterior
                        现在还是先用prior采集数据到explorer buffer,
                    不同的是,采集期间每采集200steps就更新一次posteroir,总共更新5轮.(理论上要确保posterior的质量,informative enough)
                    而在rl data collection时,从posterior中采样一个Z,将其固定住,进行rollout,不再更新posterior
                        
                    pearl是一边执行策略,一边根据收集到的信息对task belief进行后验更新.
                    然而pearl的policy不是纯粹的exploration policy,不可能一步到位得到最终的task belief,整个过程是循序渐进的,
                    如果agent在adaptation时的task belief不准确(必然的),
                    那么在后验的更新过程中,如果后验分布变化幅度足够大(有新发现,推翻之前的task belief),
                    那么之前的effort就都白费了,相当于发现自己弄错了,策略需要重新调整,甚至逆转,导致curve出现大振荡.
                    这就搞笑了,如果弄了半天都不知道自己要干嘛,那还弄个鸡巴.
                    后验更新得有个限度,不能让他一直更新下去.
                    因此用一个极限的探索策略先完成任务推断是合理的.
                    
                    而且如果最终performance不行,pearl很难trace.
                    而这样改进下来,如果最终performance不行,问题应该可以trace到exploration上面.
                    '''
                    #这么写的话是收集完了所有数据,再开始推后验
                    #如果一边收集一边推后验的话应该把下面这段写在collect_data()函数里面
                    # #prepare context for posterior update
                    # context = self.prepare_context(self.task_idx)
                    # #update posterior
                    # self.agent.infer_posterior(context)

                    if self.debug:
                        print(self.agent.z.shape)
                        print(self.agent.z)
                '''
                explorer采集k轮之后的
                '''
                # collect data with actor for RL training
                # self.env.reset_task(idx)
                print("\ncollect some trajectories for rl training")
                # sample z ~ posterior for rl agent to utilize
                self.collect_data(self.num_extra_rl_steps_posterior, exploration=False,  random_steps=0)#利用后验的z收集num_extra_rl_steps_posterior条轨迹，仅用于策略
            print("\nFinishing sample data from train tasks")
##############################################################################################
            # Sample train tasks and compute gradient updates on parameters.
            print("\nStrating Meta-training for Iteration {}".format(iteration))
            for train_step in range(self.num_train_steps_per_itr):#每轮迭代计算num_train_steps_per_itr次梯度              500x2000=1000000
                #更新explorer参数
                # self.explorer.update_parameters(memory=self.exploration_replay_buffer,batch_size=self.batch_size)
                #更新RL agent参数
                indices = np.random.choice(self.train_tasks, self.meta_batch)#train_tasks中随机取meta_batch个task , sample RL batch b~B
                if ((train_step + 1) % 500 == 0):
                    print("\nTraining step {}".format(train_step + 1))
                    print("Task indices: {}".format(indices))
                    # print("alpha:{}".format(self.alpha))
                self._do_training(indices)#梯度下降
                self._n_train_steps_total += 1

            # eval
            self._try_to_eval(iteration)#训练完了，评估模型

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        pass

    def collect_data(self, num_samples, exploration, random_steps=0):
        # start from the prior
        if exploration:
            print("explorer clear z, sample data from prior")
            '''
            每次都从prior采样
            agent.clear_z()函数
            将分布重置为标准正态分布,然后从中采样z
            '''
            self.explorer.clear_z()
        else:
            '''
            explorer在k轮更新后的z作为rl的初始z,为rl agent所利用,z之后不再改动
            '''
            print("RL agent initialize z with explorer's z")
            print("explorer z:",self.explorer.z)
            self.agent.z = self.explorer.z
            print("agent z:",self.agent.z)

        total_steps = 0
        # total_episodes = 0
        while total_steps < num_samples:
            # self.sampler.obtain_samples(max_samples=num_samples - num_transitions,max_trajs=update_posterior_rate,accum_context=False,resample=resample_z_rate)
            #采集一条轨迹
            if exploration:#用于exploration
                paths, n_steps, n_episodes= self.exploration_sampler.obtain_samples(max_samples=num_samples-total_steps,
                                                                         max_trajs=1,#只采集一条轨迹
                                                                         random_steps=random_steps)
                self.exploration_replay_buffer.add_paths(self.task_idx, paths)
                print("exploration buffer: {}, size: {}".format(self.task_idx,self.exploration_replay_buffer.task_buffers[self.task_idx].size()))


                # prepare context for posterior update
                context = self.prepare_context(self.task_idx)
                # update posterior
                if self.debug:
                    print("sample context,update posterior,then sample z from it")
                    print("old z:", self.explorer.z)
                self.explorer.infer_posterior(context)
                if self.debug:
                    print("new z:", self.explorer.z)

            else:#用于rl-training
                # paths, n_steps, n_episodes = self.sampler.obtain_samples(max_samples=num_samples - total_steps,
                #                                                          max_trajs=1,
                #                                                          accum_context=False,
                #                                                          resample=1)
                paths, n_steps, n_episodes = self.sampler.obtain_samples(
                    max_samples=num_samples - total_steps,
                    max_trajs=1,  # 只采集一条轨迹
                    random_steps=random_steps)

                self.rl_replay_buffer.add_paths(self.task_idx, paths)
                print("RL buffer: {}, size: {}".format(self.task_idx, self.rl_replay_buffer.task_buffers[self.task_idx].size()))
                #一边policy rollout一边更新后验,需要吗?不需要,因为已经花了很多时间在exploration & inference上面了.
                # 而且如果Z在训练过程中变动较大,很可能导致训练不稳定
                # context = self.prepare_context(self.task_idx)
                # self.agent.infer_posterior(context)

            total_steps += n_steps
            # total_episodes += n_episodes


            #     context = self.sample_context(self.task_idx)
            #     self.agent.infer_posterior(context)
        self._n_env_steps_total += total_steps

    def prepare_encoder_data(self, obs, act, rewards, next_obs):
        ''' prepare context for encoding '''
        # for now we embed only observations and rewards
        # assume obs and rewards are (task, batch, feat)
        task_data = torch.cat([obs, act, rewards,next_obs], dim=2)
        return task_data

    def prepare_context(self, idx):
        '''
        sample context from replay buffer and prepare it
        从exploration_replay_buffer里采样embedding_batch_size条数据,然后按照dim=2,cat起来
        '''
        debug = False
        batch = ptu.np_to_pytorch_batch(self.exploration_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent))
        obs = batch['observations'][None, ...]
        act = batch['actions'][None, ...]
        rewards = batch['rewards'][None, ...]
        next_obs = batch["next_observations"][None, ...]
        if debug:
            # print("batch:{}".format(batch))
            print(obs.shape)
            print(act.shape)
            print(rewards.shape)
            print(next_obs.shape)
        context = self.prepare_encoder_data(obs, act, rewards,next_obs)
        if debug:
            print(context.shape)
        # print(context.shape)
            # print("context:{}".format(context))
        return context

    def _try_to_eval(self, epoch):
        # logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate():
            self.evaluate(epoch)

            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            # table_keys = logger.get_table_key_set()
            # if self._old_table_keys is not None:
            #     assert table_keys == self._old_table_keys, (
            #         "Table keys cannot change from iteration to iteration."
            #     )
            # self._old_table_keys = table_keys

            logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            # logger.record_tabular(
            #     "Number of env steps total",
            #     self._n_env_steps_total,
            # )
            # logger.record_tabular(
            #     "Number of rollouts total",
            #     self._n_rollouts_total,
            # )
            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.
        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.
        :return:
        """
        # eval collects its own context, so can eval any time
        return True

    def _can_train(self):
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation, )

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    ##### Snapshotting utils #####
    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    # def collect_paths(self, idx, epoch, run):
    def collect_paths(self, idx):#收集数据进行评估
        self.task_idx = idx
        self.env.reset_task(idx)

        self.explorer.clear_z()
        paths = []
        num_transitions = 0
        num_trajs = 0
        while num_transitions < self.num_steps_per_eval:
            path, step, traj = self.sampler.obtain_samples(max_samples=self.num_steps_per_eval - num_transitions,
                                                    max_trajs=1,
                                                    random_steps=0)
            paths += path
            num_transitions += step
            num_trajs += 1
            if num_trajs >= self.num_exp_traj_eval:
                self.agent.infer_posterior(self.agent.context)

        # if self.sparse_rewards:
        #     for p in paths:
        #         sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
        #         p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal  # goal

        # save the paths for visualization, only useful for point mass
        # if self.dump_eval_paths:
        #     logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        return paths

    def _do_eval(self, indices, epoch):
        final_returns = []
        online_returns = []
        for idx in indices:#对所有任务:
            all_rets = []
            for r in range(self.num_evals):
                paths = self.collect_paths(idx)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
            final_returns.append(np.mean([a[-1] for a in all_rets]))
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0)  # avg return per nth rollout
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return final_returns, online_returns

    def evaluate(self, epoch):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        ### sample trajectories from prior for debugging / visualization
        # if self.dump_eval_paths:
        #     # 100 arbitrarily chosen for visualizations of point_robot trajectories
        #     # just want stochasticity of z, not the policy
        #     self.agent.clear_z()
        #     # prior_paths, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
        #     #                                              max_samples=self.max_path_length * 20,
        #     #                                              accum_context=False,
        #     #                                              resample=1)
        #     prior_paths, n_steps, n_episodes = self.sampler.obtain_samples(
        #         max_samples=self.max_path_length * 20,
        #         max_trajs=np.inf, #4000步,不管多少条轨迹
        #         random_steps=0)
        #     logger.save_extra_data(prior_paths, path='eval_trajectories/prior-epoch{}'.format(epoch))

        ### train tasks
        # eval on a subset of train tasks for speed
        indices = np.random.choice(self.train_tasks, len(self.eval_tasks))#从100个训练任务中随机选取30个任务进行评估(humanoid)
        print('evaluating on {} train tasks'.format(len(indices)))
        eval_util.dprint('evaluating on {} train tasks'.format(len(indices)))
        ### eval train tasks with posterior sampled from the training replay buffer
        train_returns = []
        for idx in indices:
            self.task_idx = idx
            self.env.reset_task(idx)
            paths = []
            for _ in range(self.num_steps_per_eval // self.max_path_length):#采集3次,每条轨迹不超过200
                context = self.prepare_context(idx)#exploration buffer抽出任务context
                self.explorer.infer_posterior(context)#利用context算出对应任务分布,从中采样
                self.agent.z = self.explorer.z#agent利用explorer的判断信息
                path, step, traj = self.sampler.obtain_samples(max_samples=self.max_path_length,#agent开始采样
                                                      max_trajs=1,
                                                      random_steps=0)
                paths += path

            train_returns.append(eval_util.get_average_returns(paths))
        train_returns = np.mean(train_returns)

        ### eval train tasks with on-policy data to match eval of test tasks
        train_final_returns, train_online_returns = self._do_eval(indices, epoch)
        eval_util.dprint('train online returns')
        eval_util.dprint(train_online_returns)

        ### test tasks
        eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        test_final_returns, test_online_returns = self._do_eval(self.eval_tasks, epoch)
        eval_util.dprint('test online returns')
        eval_util.dprint(test_online_returns)

        # save the final posterior
        self.agent.log_diagnostics(self.eval_statistics)

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(paths, prefix=None)

        avg_train_return = np.mean(train_final_returns)
        avg_test_return = np.mean(test_final_returns)
        avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
        avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
        self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
        self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
        self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
        logger.save_extra_data(avg_train_online_return, path='online-train-epoch{}'.format(epoch))
        logger.save_extra_data(avg_test_online_return, path='online-test-epoch{}'.format(epoch))

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()

    # @abc.abstractmethod
    # def training_mode(self, mode):
    #     """
    #     Set training mode to `mode`.
    #     :param mode: If True, training will happen (e.g. set the dropout
    #     probabilities to not all ones).
    #     """
    #     pass
    #
    # @abc.abstractmethod
    # def _do_training(self):
    #     """
    #     Perform some update, e.g. perform one gradient step.
    #     :return:
    #     """
    #     pass