import numpy as np

from rlkit.samplers.util import rollout
from rlkit.torch.sac.policies import MakeDeterministic
from sac.sacAgent import SACAgent


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
    def __init__(self, env,max_path_length):
        self.env = env
        self.agent = SACAgent(env.observation_space.shape[0], env.action_space)
        self.max_path_length = max_path_length


    def obtain_samples(self, deterministic=False, max_samples=np.inf, max_trajs=np.inf, accum_context=True, resample=1):
        """
        Obtains samples in the environment until either we reach either max_samples transitions or
        num_traj trajectories.
        The resample argument specifies how often (in trajectories) the agent will resample it's context.
        """
        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        policy = MakeDeterministic(self.policy) if deterministic else self.policy
        paths = []
        n_steps_total = 0
        n_trajs = 0
        while n_steps_total < max_samples and n_trajs < max_trajs:
            path = rollout(
                self.env, policy, max_path_length=self.max_path_length, accum_context=accum_context)
            # save the latent context that generated this trajectory
            path['context'] = policy.z.detach().cpu().numpy()
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1
            # don't we also want the option to resample z ever transition?
            if n_trajs % resample == 0:
                policy.sample_z()
        return paths, n_steps_total

