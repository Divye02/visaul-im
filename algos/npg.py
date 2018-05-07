import logging
logging.disable(logging.CRITICAL)
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA
import copy
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
# samplers
import visual_im.samplers.trajectory_sampler as trajectory_sampler
import visual_im.samplers.batch_sampler as batch_sampler
from multiprocessing import Pool

# utility functions
import visual_im.utils.process_samples as process_samples
from visual_im.utils.logger import DataLog


class NaturalPolicyGradients:
    def __init__(self, env, policy, baseline,
                 delta=0.1,
                 seed=None,
                 save_logs=False):

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.seed = seed
        self.save_logs = save_logs
        self.delta = delta
        self.alpha = 1
        self.running_score = None
        np.random.seed(seed)
        if save_logs: self.logger = DataLog()

    def CPI_surrogate(self, observations, actions, advantages):
        advantages = advantages / (np.max(advantages) + 1e-8)
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False)
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        LR = self.policy.likelihood_ratio(new_dist_info, old_dist_info)
        surr = torch.mean(LR*adv_var)
        return surr

    def kl_old_new(self, observations, actions):
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        mean_kl = self.policy.mean_kl(new_dist_info, old_dist_info)
        return mean_kl

    def flat_vpg(self, observations, actions, advantages):
        cpi_surr = self.CPI_surrogate(observations, actions, advantages)
        vpg_grad = torch.autograd.grad(cpi_surr, self.policy.trainable_params)
        vpg_grad = np.concatenate([g.contiguous().view(-1).data.numpy() for g in vpg_grad])
        return vpg_grad

    # ----------------------------------------------------------
    def train_step(self, N,
                   sample_mode='trajectories',
                   domain_name=None,
                   task_name=None,
                   T=1e6,
                   gamma=0.995,
                   gae_lambda=0.98,
                   num_cpu='max'):

        # Clean up input arguments
        if domain_name is None or task_name is None: domain_name = self.env.domain_name; task_name = self.env.task_name
        if sample_mode is not 'trajectories' and sample_mode is not 'samples':
            print("sample_mode in NPG must be either 'trajectories' or 'samples'")
            quit()

        ts = timer.time()

        if sample_mode is 'trajectories':
            paths = trajectory_sampler.sample_paths_parallel(N, self.policy, T, domain_name, task_name,
                                                             self.seed, num_cpu)
        elif sample_mode is 'samples':
            paths = batch_sampler.sample_paths(N, self.policy, T, domain_name=domain_name, task_name=task_name,
                                               pegasus_seed=self.seed, num_cpu=num_cpu)

        if self.save_logs:
            self.logger.log_kv('time_sampling', timer.time() - ts)

        self.seed = self.seed + N if self.seed is not None else self.seed

        # compute returns
        process_samples.compute_returns(paths, gamma)
        # compute advantages
        process_samples.compute_advantages(paths, self.baseline, gamma, gae_lambda)
        # train from paths
        eval_statistics = self.train_from_paths(paths)
        eval_statistics.append(N)
        # fit baseline
        if self.save_logs:
            ts = timer.time()
            error_before, error_after = self.baseline.fit(paths, return_errors=True)
            self.logger.log_kv('time_VF', timer.time()-ts)
            self.logger.log_kv('VF_error_before', error_before)
            self.logger.log_kv('VF_error_after', error_after)
        else:
            self.baseline.fit(paths)

        return eval_statistics

    # ----------------------------------------------------------
    def train_from_paths(self, paths):

        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        self.running_score = mean_return if self.running_score is None else \
                             0.9*self.running_score + 0.1*mean_return  # approx avg of last 10 iters
        if self.save_logs: self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_opt = 0.0

        # Optimization algorithm
        # --------------------------
        ts = timer.time()
        surr_before = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        curr_params = self.policy.get_param_values()
        vpg_grad = self.flat_vpg(observations, actions, advantages)

        F = self.get_fisher_mat(observations, actions, sub_s = 0.1)
        F[np.diag_indices_from(F)] += 1e-8
        npg_grad = spLA.cg(F, vpg_grad, maxiter=15)[0]
        self.alpha = self.calculate_alpha(vpg_grad, npg_grad)

        # print("alpha", alpha)
        # print("npg grad", npg_grad)
        # print("vpg grad", vpg_grad)
        new_params, new_surr, kl_dist = self.simple_gradient_update(curr_params, npg_grad, self.alpha,
                                        observations, actions, advantages)


        self.policy.set_param_values(new_params, set_new=True, set_old=True)
        surr_improvement = new_surr - surr_before
        t_opt += timer.time() - ts

        # Log information
        if self.save_logs:
            self.logger.log_kv('time_opt', t_opt)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('alpha', self.alpha)
            self.logger.log_kv('surr_improvement', surr_improvement)
            self.logger.log_kv('running_score', self.running_score)

        return base_stats

    def calculate_alpha(self, vpg_grad, npg_grad):
        alpha = (self.delta / (np.dot(vpg_grad, npg_grad))) ** 0.5
        return alpha

    def get_fisher_mat(self, observations, actions, sub_s=0.1):
        sub_i = np.random.choice(observations.shape[0], int(sub_s * observations.shape[0]))
        ll = self.policy.new_dist_info(observations[sub_i], actions[sub_i])[0]
        # ll = torch.sort(ll, descending=False)[0]

        # sub_i = np.random.choice(int(len(observations)*0.7), int(0.1 * observations.shape[0]))

        # ll = ll[torch.LongTensor(sub_i)]
        F = np.zeros(1)
        for ll_i in ll:
            grad_pol = torch.autograd.grad(ll_i, self.policy.trainable_params, retain_graph=True)
            grad_pol = np.concatenate([g.contiguous().view(-1).data.numpy() for g in grad_pol])
            F_i = np.dot(grad_pol.reshape(-1,1), grad_pol.reshape(1,-1))
            F = F + F_i

        # print("F", F)
        return F / len(ll)

        # ll = self.policy.new_dist_info(observations, actions)[0]
        # all_g = np.array(list(map(lambda ele: np.concatenate([g.contiguous().view(-1).data.numpy() for g in
        #                                                       torch.autograd.grad(ele, self.policy.trainable_params,
        #                                                                           retain_graph=True)]), ll)))
        # all_g = np.expand_dims(all_g, axis=2)
        # F = np.mean(np.dot(all_g, all_g.transpose([0,2,1])), axis=0)
        # return F



    def log_rollout_statistics(self, paths):
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        self.logger.log_kv('stoc_pol_mean', mean_return)
        self.logger.log_kv('stoc_pol_std', std_return)
        self.logger.log_kv('stoc_pol_max', max_return)
        self.logger.log_kv('stoc_pol_min', min_return)

    def simple_gradient_update(self, curr_params, search_direction, step_size,
                               observations, actions, advantages):
        # This function takes in the current parameters, a search direction, and a step size
        # and computes the new_params =  curr_params + step_size * search_direction.
        # It also computes the CPI surrogate at the new parameter values.
        # This function also computes KL(pi_new || pi_old) as discussed in the class,
        # where pi_old = policy with current parameters (i.e. before any update),
        # and pi_new = policy with parameters equal to the new_params as described above.
        # The function DOES NOT set the parameters to the new_params -- this has to be
        # done explicitly outside this function.

        new_params = curr_params + step_size*search_direction
        self.policy.set_param_values(new_params, set_new=True, set_old=False)
        new_surr = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(curr_params, set_new=True, set_old=True)
        return new_params, new_surr, kl_dist
