import numpy as np
from dm_control import suite
from dm_control.rl.environment import StepType
from visual_im.visual.viewer import DmControlViewer

class EnvSpec(object):
    def __init__(self, obs_dim, act_dim, horizon, num_agents):
        self.observation_dim = obs_dim
        self.action_dim = act_dim
        self.horizon = horizon
        self.num_agents = num_agents

class DeepMindEnv(object):
    def __init__(self, domain_name, task_name):
        env = suite.load(domain_name=domain_name, task_name=task_name)
        self.env = env
        self.task_name = task_name
        self.domain_name = domain_name
        self._horizon = 500

        self._action_dim = self.env.action_spec().shape[0]

        self._observation_dim = int(np.sum([v.shape[0] for k, v in env.observation_spec().items()]))

        self._num_agents = 1

        # Specs
        self.spec = EnvSpec(self._observation_dim, self._action_dim, self._horizon, self._num_agents)

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def observation_dim(self):
        return self._observation_dim

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.physics.render()

    def visualize_policy(self, policy, horizon=1000, num_episodes=1, mode='exploration'):
        pixels = self.env.physics.render()
        self.renderer = DmControlViewer(pixels.shape[1], pixels.shape[0])
        for ep in range(num_episodes):
            o = self.reset()
            d = False
            t = 0

            o = np.concatenate(list(o.observation.values()))

            while t < horizon and d is False:
                pixels = self.env.physics.render()
                self.renderer.update(pixels)
                a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
                step_type, r, discount, next_o = self.step(a)
                d = step_type == StepType.LAST
                t = t+1


        self.mujoco_render_frames = False


    def seed(self, seed):
        self.env.task.random.seed(seed)

    def evaluate_policy(self, policy, 
                        num_episodes=5, 
                        horizon=None, 
                        gamma=1, 
                        visual=False,
                        percentile=[], 
                        get_full_dist=False, 
                        mean_action=False,
                        terminate_at_done=True,
                        save_video_location=None,
                        seed=None):

        if seed is not None:
            self.env.task.random.seed(seed)
        horizon = self._horizon if horizon is None else horizon
        mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
        ep_returns = np.zeros(num_episodes)

        # if save_video_location != None:
        #     self.env.monitor.start(save_video_location, force=True)

        for ep in range(num_episodes):


            o = self.reset()

            t, done = 0, False
            while t < horizon and (done == False or terminate_at_done == False):
                if visual == True:
                    self.render()
                if mean_action:
                    a = policy.get_action(o)[1]['mean']
                else:
                    a = policy.get_action(o)[0]
                o, r, done, _ = self.step(a)
                ep_returns[ep] += (gamma ** t) * r
                t += 1
        
        # if save_video_location != None:
        #     self.env.monitor.close()
        
        mean_eval, std = np.mean(ep_returns), np.std(ep_returns)
        min_eval, max_eval = np.amin(ep_returns), np.amax(ep_returns)
        base_stats = [mean_eval, std, min_eval, max_eval]

        percentile_stats = []
        full_dist = []

        for p in percentile:
            percentile_stats.append(np.percentile(ep_returns, p))

        if get_full_dist == True:
            full_dist = ep_returns

        return [base_stats, percentile_stats, full_dist]
