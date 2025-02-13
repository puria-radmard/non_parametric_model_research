# Take synthetic data generated from models, and the real data, and histogram their raw errors
# Then, remove the best fitting uniform + 

import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch import Tensor as _T

from purias_utils.error_modelling_torus.data_utils.loading_utils import load_experimental_data
from purias_utils.util.arguments_yaml import ConfigNamepace
from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model.main import NonParametricSwapErrorsGenerativeModel
from purias_utils.error_modelling_torus.non_parametric_error_model.main import WorkingMemorySimpleSwapModel


results_base = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_single_subject_summaries_7_10_24"
figure_base = "/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/figures"


# dataset_name = 'mcmaster2022_e2_dircue'
# num_subjects = 10
# set_sizes = [4]
# stim_conditions = ['_lowC', '_medC', '_highC']

dataset_name = 'mcmaster2022_e1_oricue'
num_subjects = 10
set_sizes = [6]
stim_conditions = ['_cue_AR1', '_cue_AR2', '_cue_AR3']


model_names = [
    'spike_and_slab_von_mises_and_uniform',
    'spike_and_slab_wrapped_stable',
    'cue_dim_von_mises_and_uniform',
    'cue_dim_wrapped_stable',
    'est_dim_von_mises_and_uniform',
    'est_dim_wrapped_stable',
    'full_von_mises_and_uniform',
    'full_wrapped_stable',
]




assert (len(set_sizes) == 1 or len(stim_conditions) == 1)
if len(stim_conditions) == 1:
    assert stim_conditions[0] == ''
    num_fig_cols = len(set_sizes)
else:
    num_fig_cols = len(stim_conditions)



real_raw_errors = {(ss, cond, sidx): None for ss in set_sizes for cond in stim_conditions for sidx in range(num_subjects)}
all_synth_errors = {}


######
###### Collect real and synthetic errors
######

for stim_condition in stim_conditions:

    for model_name in model_names:

        for subject_index in range(num_subjects):

            log_base = os.path.join(results_base, f"{dataset_name}{stim_condition}", f"{model_name}_{subject_index}")
            results_path = os.path.join(log_base, "recent_losses.npy")
            config_path = os.path.join(log_base, "args.yaml")
            synth_data_path = os.path.join(log_base, "synthetic_data_single_subject_cross_fit.npy")
            log_path = os.path.join(log_base, "epoch_log_train.csv")

            try:
                synthetic_data = np.load(synth_data_path, allow_pickle=True).item()
                num_lines = sum(1 for _ in open(log_path))
                assert num_lines > 10
            except (FileNotFoundError, AssertionError):
                print('MISSING:', log_base)
                continue

            # Just do this one time, using args above
            if any([v is None for k, v in real_raw_errors.items() if k[2] == subject_index]):
                recreated_args = ConfigNamepace.from_yaml_path(config_path)
                real_dataset = load_experimental_data(
                    recreated_args.dataset_name,
                    recreated_args.train_indices_seed,
                    recreated_args.train_indices_path,
                    recreated_args.M_batch,
                    recreated_args.M_test_per_set_size, 
                    recreated_args.num_models,
                    recreated_args
                )

            for set_size in set_sizes:

                if real_raw_errors[(set_size, stim_condition, subject_index)] is None:
                    new_raw_errors = real_dataset.data_generators[set_size].all_errors[0,:,0].cpu().numpy()            # 
                    real_raw_errors[(set_size, stim_condition, subject_index)] = new_raw_errors # np.histogram(real_raw_errors, bins = histogram_bin_lower_bounds, density = True)[0]

                new_synth_errors = synthetic_data['generated_data']['errors'][set_size][0,:,0]    # XXX: only taking one set of synthetic parameters!!!
                all_synth_errors[(set_size, stim_condition, subject_index, model_name)] = new_synth_errors # np.histogram(new_synth_errors, bins = histogram_bin_lower_bounds, density = True)[0]



######
###### Now stack them and keep track of what the order is
######

all_labels = []
all_errors = []
for k, v in real_raw_errors.items():
    if v is None:
        continue
    all_labels.append(k)
    all_errors.append(v)


for k, v in all_synth_errors.items():
    if v is None:
        continue
    all_labels.append(k)
    all_errors.append(v)


all_errors = torch.tensor(np.concatenate(all_errors, 0)).cuda()          # [Many, M per subject, 1]



######
###### Now fit a massive N=1 spike and slab von Mises + uniform model to it
######

big_ol_vm_and_unif_model = WorkingMemorySimpleSwapModel(generative_model = NonParametricSwapErrorsGenerativeModel.from_typical_args(
    num_models = len(all_errors),
    swap_type = 'spike_and_slab',
    kernel_type = None,
    emission_type = 'von_mises',
    fix_non_swap = False,
    remove_uniform = False,
    include_pi_u_tilde = True,
    include_pi_1_tilde = False,
    normalisation_inner = 'exp',
    shared_swap_function = False,
    shared_emission_distribution = False,
    all_set_sizes = [1],
    trainable_kernel_delta = False,
    num_features = 0,
    device = 'cuda',
)).cuda()
optim = torch.optim.Adam(big_ol_vm_and_unif_model.parameters(), lr = 0.01)


timesteps = 5000
all_losses = np.zeros([timesteps, len(all_errors)])


for t in tqdm(range(timesteps)):

    big_ol_vm_and_unif_model.zero_grad()

    losses = big_ol_vm_and_unif_model.get_elbo_terms_easier(all_errors, all_errors.shape[1], 1, 'error')
    (- losses['total_log_likelihood']).mean().backward()
    optim.step()

    all_losses[t,:] = losses['total_log_likelihood'].detach().cpu().numpy()

    if t % 100 == 0:
        plt.close('all')
        fig_loss, ax_loss = plt.subplots(1)
        for llll in all_losses[:t].T:
            ax_loss.plot(llll)

        fig_loss.savefig(os.path.join(figure_base, f'training_loss_for_raw_error_histogram_removal_{dataset_name}.png'))



######
###### Evaluate the llh pdf of these vm + unif mixtures on a grid
######

histogram_bin_lower_bounds = np.linspace(-np.pi, +np.pi, 21)
histogram_bin_mid_points = 0.5 * (histogram_bin_lower_bounds[:-1] + histogram_bin_lower_bounds[1:])

canvas = torch.linspace(-torch.pi, +torch.pi, 721)
dwidth = (canvas[1] - canvas[0]).item()
canvas_mid_points = 0.5 * (canvas[:-1] + canvas[1:])
canvas_mid_points = canvas_mid_points.unsqueeze(0).repeat(len(all_errors), 1).cuda().unsqueeze(-1)

with torch.no_grad(): 
    componentwise_pdfs = big_ol_vm_and_unif_model.generative_model.error_emissions.individual_component_likelihoods_from_estimate_deviations(1, canvas_mid_points).cpu().numpy()     # [many, 720]
    fitted_vm_and_unif_weighs = big_ol_vm_and_unif_model.generative_model.swap_function.generate_pi_vectors(1, 720)['pis'].cpu().numpy()
    best_fit_pdfs = (componentwise_pdfs * fitted_vm_and_unif_weighs).sum(-1)


canvas_mid_points = canvas_mid_points.cpu().numpy().squeeze(-1)

canvas_agg = np.zeros([len(all_errors), 20])
for agg_idx, (lower, upper) in enumerate(zip(histogram_bin_lower_bounds[:-1], histogram_bin_lower_bounds[1:])):
    agg_index_mask = np.logical_and(lower <= canvas_mid_points[0], canvas_mid_points[0] < upper)
    width = upper - lower
    import pdb; pdb.set_trace()
    canvas_agg[:,agg_idx] = best_fit_pdfs[:,agg_index_mask].sum(-1) * dwidth / width


print(np.abs(canvas_agg.sum(-1) * width - 1.0).max())




all_errors = all_errors.cpu().numpy()

#### XXX
fig, axes = plt.subplots(2, num_fig_cols, figsize = (10 * num_fig_cols, 20))    # Top row = raw, bottom row = raw minus best fitting

example_idx = -40
new_label = all_labels[example_idx]
new_errors = all_errors[example_idx,:,0]
new_aggregated_bestfit_pdf = canvas_agg[example_idx]
new_hist_heights = np.histogram(new_errors, histogram_bin_lower_bounds, density = True)[0]

new_removed_hist_heights = new_hist_heights - new_aggregated_bestfit_pdf

axes[0,0].plot(histogram_bin_mid_points, new_hist_heights)
axes[0,0].plot(histogram_bin_mid_points, new_aggregated_bestfit_pdf)
axes[0,0].plot(histogram_bin_mid_points, new_removed_hist_heights)

fig.savefig(os.path.join(figure_base, f'raw_error_histogram_removal_{dataset_name}.png'))

import pdb; pdb.set_trace()
#### XXX
pass

