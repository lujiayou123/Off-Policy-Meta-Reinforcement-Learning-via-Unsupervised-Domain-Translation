import numpy as np

from rlkit.samplers.util import rollout
from rlkit.torch.sac.policies import MakeDeterministic


class InPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_path_length):
        self.env = env
        self.policy = policy

        self.max_path_length = max_path_length

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self, deterministic=False, max_samples=np.inf, max_trajs=np.inf, accum_context=True, resample=1):
        """
        Obtains samples in the environment until either we reach either max_samples transitions or
        num_traj trajectories.
        The resample argument specifies how often (in trajectories) the agent will resample it's context.
        """
        # assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        # policy = MakeDeterministic(self.policy) if deterministic else self.policy
        policy = self.policy
        paths = []
        n_steps_total = 0
        n_trajs = 0
        while n_steps_total < max_samples and n_trajs < max_trajs:
            path = self.rollout(
                env=self.env, agent=policy, max_path_length=self.max_path_length, random_steps=0)
            # save the latent context that generated this trajectory
            # path['context'] = policy.z.detach().cpu().numpy()
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1
            # don't we also want the option to resample z ever transition?
            # if n_trajs % resample == 0:
            #     policy.sample_z()
        return paths, n_steps_total, n_trajs

    def rollout(env, agent, max_path_length, random_steps=0):
        observations = []
        actions = []
        rewards = []
        terminals = []
        next_observations = []
        o = env.reset()
        path_length = 0

        while path_length < max_path_length:

            if path_length < random_steps:
                print("random step")
                a = env.action_space.sample()
            else:
                a, agent_info = agent.get_action(o)  # PEARLAgent.get_action(),return tuple
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


