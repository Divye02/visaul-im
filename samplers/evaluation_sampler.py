import logging

logging.disable(logging.CRITICAL)

import numpy as np
from visual_im.utils.get_environment import get_environment
from visual_im.utils import tensor_utils
from dm_control.rl.environment import StepType

# Single core rollout to sample trajectories
# =======================================================
def do_evaluation_rollout(N,
                          policy,
                          T=1e6,
                          env=None,
                          domain_name=None,
                          task_name=None,
                          pegasus_seed=None):
    """
    params:
    N               : number of trajectories
    policy          : policy to be used to sample the data
    T               : maximum length of trajectory
    env             : env object to sample from
    env_name        : name of env to be sampled from
                      (one of env or env_name must be specified)
    pegasus_seed    : seed for environment (numpy speed must be set externally)
    """

    if (domain_name is None or task_name is None) and env is None:
        print("No environment specified! Error will be raised")
    if env is None: env = get_environment(domain_name, task_name)
    if pegasus_seed is not None: env.env._seed(pegasus_seed)
    T = min(T, env.horizon)

    # print("####### Worker started #######")

    paths = []

    for ep in range(N):

        # Set pegasus seed if asked
        if pegasus_seed is not None:
            seed = pegasus_seed + ep
            env.env._seed(seed)
            np.random.seed(seed)
        else:
            np.random.seed()

        observations = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []

        o = env.reset()
        done = False
        t = 0

        o = np.concatenate(list(o.observation.values()))
        while t < T and done != True:
            _, agent_info = policy.get_action(o)
            a = agent_info['evaluation']
            step_type, r, discount, next_o = env.step(a)
            done = step_type == StepType.LAST
            # observations.append(o.ravel())
            observations.append(o)
            actions.append(a)
            rewards.append(r)
            agent_infos.append(agent_info)
            # env_infos.append(env_info)
            o = np.concatenate(list(next_o.values()))
            t += 1

        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            # env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            terminated=done
        )

        paths.append(path)

    # print("====== Worker finished ======")

    return paths


def do_evaluation_rollout_star(args_list):
    return do_evaluation_rollout(*args_list)
