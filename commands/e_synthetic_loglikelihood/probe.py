# Just double checking that synthetic data generation has gone generally right!

import torch
import numpy as np
import matplotlib.pyplot as plt

synth_data_path = 'results_link/e_synthetic_likelihood_24_9_24/trained_on_data/mcmaster2022_e1_oricue_cue_AR2/spike_and_slab_von_mises_and_uniform/run_0/synthetic_data_likelihood_dist.npy'
parameters_path = 'results_link/e_synthetic_likelihood_24_9_24/trained_on_data/mcmaster2022_e1_oricue_cue_AR2/spike_and_slab_von_mises_and_uniform/run_0/generative_model.mdl'

synth_data = np.load(synth_data_path, allow_pickle=True).item()

set_size = 6

components = synth_data['generated_data']['components'][set_size].astype(int)  # [K, M]
errors = synth_data['generated_data']['errors'][set_size]                      # [K, M, N]
K, M, N = errors.shape


all_errors = []

for k in range(K):
    for m in range(M):
        n = components[k,m]
        if n == 0:
            all_errors.append(errors[k,m,0])
        else:
            all_errors.append(errors[k,m,n-1])

params = torch.load(parameters_path)

pi_swap_tilde = params[f'swap_function.pi_swap_tilde_holder.{set_size}.pi_tilde']
pi_u_tilde = params[f'swap_function.pi_u_tilde_holder.{set_size}.pi_tilde']
pi_non_swap_tilde = 1.0

full_logit_vector = torch.tensor([pi_u_tilde, pi_non_swap_tilde] + [pi_swap_tilde for _ in range(set_size - 1)])

pi_u, pi_non_swap, pi_each_swap, *_ = full_logit_vector.softmax(0).tolist()
prior_total_pi_swap = pi_each_swap * (set_size - 1)

total_von_mises_weight = prior_total_pi_swap + pi_non_swap

grid = torch.linspace(-torch.pi, +torch.pi, 361)
von_mises = torch.distributions.VonMises(loc = 0.0, concentration=params[f'error_emissions.concentration_holder.{set_size}.log_concentration'].exp().cpu()).log_prob(grid).exp()
von_mises_and_uniform = (total_von_mises_weight * von_mises) + (pi_u / 2 / np.pi)

plt.plot(grid, von_mises_and_uniform)

plt.hist(all_errors, bins=100, density = True)

emp_pi_u = (components == 0).sum() / components.size
emp_pi_non_swap = (components == 1).sum() / components.size
emp_total_pi_swap = (components > 1).sum() / components.size

plt.suptitle(
    f'Priors (ns/s/u): {round(pi_non_swap, 3), round(prior_total_pi_swap, 3), round(pi_u, 3)}\n'
    f'Empirical (ns/s/u): {round(emp_pi_non_swap, 3), round(emp_total_pi_swap, 3), round(emp_pi_u, 3)}'
)


plt.savefig('/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/e_synthetic_loglikelihood/probe.png')
