import os, torch, json
import numpy as np
from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt

from purias_utils.util.arguments_yaml import ConfigNamepace

from purias_utils.error_modelling_torus.data_utils.loading_utils import load_experimental_data
from purias_utils.error_modelling_torus.non_parametric_error_model.setup_utils import setup_model_whole

from non_parametric_model.scripts.main.visualisation_utils import inference_on_grid
from purias_utils.error_modelling_torus.non_parametric_error_model.util import get_elbo_terms

########################################

dataset = 'mcmaster2022_e1_loccue'
emission_type = 'wrapped_stable'
bnp_type = 'cue_dim'
set_size = 6

########################################

config_yaml_base_path = "/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data"

config_yaml_path = os.path.join(config_yaml_base_path, f"{dataset}_{bnp_type}_{emission_type}.yaml")
config_yaml = ConfigNamepace.from_yaml_path(config_yaml_path)
logging_base = config_yaml.logging_base

baseline_config_yaml_path = os.path.join(config_yaml_base_path, f"{dataset}_spike_and_slab_{emission_type}.yaml")
baseline_config_yaml = ConfigNamepace.from_yaml_path(baseline_config_yaml_path)
baseline_logging_base = baseline_config_yaml.logging_base


all_set_sizes = [set_size]

result_dest_path = f"/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/figures/model_comparison/{dataset}/{bnp_type}_{emission_type}_summary.png"

fig, axes = plt.subplots(2, 2)

axes[1,0].set_title('Changes in marginal correct prior')
axes[1,0].set_xlabel('SnS')
axes[1,0].set_ylabel('BnP')
axes[1,1].set_title('Changes in marginal swap prior - ')
axes[1,1].set_xlabel('SnS')
axes[1,1].set_ylabel('BnP')

min_sep = 3.14

seed_to_bnp_corr_prob = {}
seed_to_bnp_swap_prob = {}
seed_to_baseline_corr_prob = {}
seed_to_baseline_swap_prob = {}

for run_path in tqdm(glob(os.path.join(baseline_logging_base, "run_*"))):
    
    run_args_path = os.path.join(run_path, "args.yaml")
    run_args = ConfigNamepace.from_yaml_path(run_args_path)
    training_set_seed = run_args.train_indices_seed

    baseline_generative_model, _, _, _ = setup_model_whole(
        run_args.swap_type, run_args.emission_type, 
        all_set_sizes, run_args.remove_uniform, run_args.include_pi_u_tilde, False, 
        run_args.R_per_dim, run_args.fix_non_swap, run_args.include_pi_1_tilde,
        run_args.fix_inducing_point_locations,
        run_args.shared_swap_function, run_args.shared_emission_distribution,
        device = 'cuda'
    )
    baseline_generative_model.load_state_dict(torch.load(os.path.join(run_path, f'generative_model.mdl')))
    baseline_prior = baseline_generative_model.swap_function.generate_pi_vectors(set_size, 1).flatten().detach().cpu()
    baseline_uniform_prob = baseline_prior[0].item()
    baseline_non_swap_prob = baseline_prior[1].item()
    baseline_swap_prob = baseline_prior[2:].sum().item()

    seed_to_baseline_corr_prob[training_set_seed] = baseline_non_swap_prob
    seed_to_baseline_swap_prob[training_set_seed] = baseline_swap_prob


for run_path in tqdm(glob(os.path.join(logging_base, "run_*"))):

    run_args_path = os.path.join(run_path, "args.yaml")
    run_args = ConfigNamepace.from_yaml_path(run_args_path)
    training_set_seed = run_args.train_indices_seed

    generative_model, variational_models, _, D, delta_dimensions = setup_model_whole(
        run_args.swap_type, run_args.emission_type, 
        all_set_sizes, run_args.remove_uniform, run_args.include_pi_u_tilde, False, 
        run_args.R_per_dim, run_args.fix_non_swap, run_args.include_pi_1_tilde,
        run_args.fix_inducing_point_locations, 
        run_args.shared_swap_function, run_args.shared_emission_distribution,
        device = 'cuda'
    )

    dataset_generator = load_experimental_data(run_args.dataset_name, run_args.train_indices_seed, run_args.train_indices_path, run_args.M_batch, run_args.M_test_per_set_size, run_args)
    with open(os.path.join(run_path, f"train_indices.json"), 'r') as jf:
        train_indices = json.load(jf)
    dg = dataset_generator.data_generators[set_size]
    dg.set_train_indices(train_indices[str(set_size)])
    all_deltas = dg.all_deltas[train_indices[str(set_size)]].detach()
    all_errors = dg.all_errors[train_indices[str(set_size)]].detach()
    all_relevant_deltas = all_deltas[...,delta_dimensions]
    own_min_sep = all_deltas[:,1:].abs().min().item()
    min_sep = min(min_sep, own_min_sep)

    parameter_load_path = os.path.join(run_path, '{model}.{ext}')
    generative_model.load_state_dict(torch.load(parameter_load_path.format(model = f'generative_model', ext = 'mdl')))
    for k, v in variational_models.items():
        v.load_state_dict(torch.load(parameter_load_path.format(model = f'variational_model_{k}', ext = 'mdl')))

    variational_model = variational_models[set_size]

    _, _, learned_upper_error_surface, learned_lower_error_surface, learned_full_grid, learned_surface, _, _, _ = inference_on_grid(
        run_args.swap_type, set_size, generative_model, variational_model, 120
    )

    with torch.no_grad():
        emissions_distribution = generative_model.error_emissions.individual_component_log_likelihoods_from_estimate_deviations_inner(
            set_size, learned_full_grid.cuda()
        ).exp().cpu().numpy()

        _, _, _, component_priors = get_elbo_terms(
            variational_model, generative_model, all_relevant_deltas, all_errors,
            len(all_deltas), set_size, 128, 'error', False, True
        )

        marginal_priors = component_priors.mean(0).mean(0)
        uniform_prob = marginal_priors[0].item()
        non_swap_prob = marginal_priors[1].item()
        swap_prob = marginal_priors[2:].sum().item()
        seed_to_bnp_corr_prob[training_set_seed] = non_swap_prob
        seed_to_bnp_swap_prob[training_set_seed] = swap_prob
    
    exp_learned_surface = np.exp(learned_surface.flatten())
    exp_learned_surface[learned_full_grid.cpu().abs() < own_min_sep] = np.nan

    axes[0,0].plot(learned_full_grid.cpu().numpy(), exp_learned_surface)
    axes[0,1].plot(learned_full_grid.cpu().numpy(), emissions_distribution)


trs = seed_to_baseline_corr_prob.keys()
baseline_corr_probs = [seed_to_baseline_corr_prob[k] for k in trs]
bnp_corr_probs = [seed_to_bnp_corr_prob.get(k, torch.nan) for k in trs]
baseline_swap_probs = [seed_to_baseline_swap_prob.get(k, torch.nan) for k in trs]
bnp_swap_probs = [seed_to_bnp_swap_prob.get(k, torch.nan) for k in trs]
axes[1,0].scatter(baseline_corr_probs, bnp_corr_probs)
axes[1,1].scatter(baseline_swap_probs, bnp_swap_probs)

for ax in axes[1]:
    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    ax.plot([min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])], [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])])
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

y_lim = axes[0,0].get_ylim()
axes[0,0].plot([+min_sep, +min_sep], y_lim, color = 'gray', linestyle = '--')
axes[0,0].plot([-min_sep, -min_sep], y_lim, color = 'gray', linestyle = '--')
axes[0,0].set_ylim(y_lim)


cued_dim_name = dataset_generator.feature_names[delta_dimensions[0]]
estimated_dim_name = dataset_generator.feature_names[1]

axes[0,0].set_title(f'Learned function mean against {cued_dim_name}')
axes[0,0].set_xlabel(cued_dim_name)
axes[0,0].set_ylabel('$e^f$')

axes[0,1].set_title(f'Learned emission distribution against {estimated_dim_name}')
axes[0,1].set_xlabel(estimated_dim_name)
axes[0,1].set_ylabel('$p$')

fig.savefig(result_dest_path)

