raise Exception('Redo - this can be parallelised with Q axis! i.e. set num models to something larger augment as needed')

# Load up a model and its training data, then go over and generate data from mean f but with a different emissions distribution

import numpy as np
import os, torch, argparse

from tqdm import tqdm

from typing import Dict

from matplotlib import pyplot as plt

from torch import Tensor as _T

# from non_parametric_model.scripts.main.setup import *
from purias_utils.util.arguments_yaml import ConfigNamepace
from purias_utils.error_modelling_torus.non_parametric_error_model.setup_utils import setup_model_whole
from purias_utils.error_modelling_torus.data_utils.loading_utils import load_experimental_data

from purias_utils.maths.circular_statistics import kurtosis_from_angles, mean_resultant_length_from_angles
from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles
from purias_utils.error_modelling_torus.non_parametric_error_model.util import get_elbo_terms, get_elbo_terms_spike_and_slab

from non_parametric_model.scripts.synthetic_data_generation import utils

from purias_utils.maths.circular_statistics import wrapped_stable_kurtosis, wrapped_stable_mean_resultant_length, step_along_mean_resultant_length_contour, step_along_kurtosis_contour, symmetric_zero_mean_wrapped_stable

from purias_utils.error_modelling_torus.non_parametric_error_model.util import inference_mean_only


parser = argparse.ArgumentParser()
parser.add_argument('--resume_path', type = str)
parser.add_argument('--synthetic_data_code', type = str, required = False, default = 'circular_statistics_sweep')
parser.add_argument('--num_contour_params', type = int, required = False, default = 3)#16)
parser.add_argument('--make_spike_and_slab', required = False, default = False, action = 'store_true')
parser.add_argument('--num_synthetic_generation_repeats_per_emission_distribution', type = int, required = False, default = 2)# 10)
parser.add_argument('--generation_source', type = str, required = False, help = "i.e. only do inference with a sweep over emission parameters here - gather the data from elsewhere...")
data_gen_args = parser.parse_args()
resume_path = data_gen_args.resume_path
args = ConfigNamepace.from_yaml_path(os.path.join(resume_path, "args.yaml"))

args.dict.pop('resume_path')
dataset_generator = load_experimental_data(args.dataset_name, args.train_indices_seed, args.train_indices_path, args.M_batch, args.M_test_per_set_size, args)
all_set_sizes = list(dataset_generator.data_generators.keys())

generative_model, variational_models, variational_model, D, delta_dimensions = setup_model_whole(
    **args.dict, all_set_sizes=all_set_sizes, trainable_kernel_delta=False, min_seps=None, resume_path=resume_path
)

generative_model.error_emissions.p_cut_off = 1000

data_destination = os.path.join(resume_path, f'synthetic_data_{data_gen_args.synthetic_data_code}.npy')
figure_destination = os.path.join(resume_path, f'synthetic_data_f_mean_evaluated_for_{data_gen_args.synthetic_data_code}.png')
config_yaml_destination = os.path.join(resume_path, f'synthetic_data_args_{data_gen_args.synthetic_data_code}.yaml')

assert resume_path is not None


#for dest in [data_destination, figure_destination, config_yaml_destination]:
#   if os.path.exists(dest):
#       raise Exception('Cannot overwrite data! ' + dest)


assert args.emission_type == 'wrapped_stable'

if data_gen_args.make_spike_and_slab:
    assert args.swap_type != 'spike_and_slab'


original_alphas = {N: generative_model.error_emissions.emission_parameter(N)[0].item() for N in all_set_sizes}
original_gammas = {N: generative_model.error_emissions.emission_parameter(N)[1].item() for N in all_set_sizes}
original_emission_first_resultant_vector_lengths = {N: wrapped_stable_mean_resultant_length(original_alphas[N], original_gammas[N]) for N in all_set_sizes}
original_emission_circular_kurtosis = {N: wrapped_stable_kurtosis(original_alphas[N], original_gammas[N]) for N in all_set_sizes}

augmentation_alphas = {}
augmentation_gammas = {}
sweep_emission_first_resultant_vector_lengths = {}
sweep_emission_circular_kurtosis = {}

fig, axes = plt.subplots(len(all_set_sizes), 5, figsize = (25, 5*len(all_set_sizes)))
if len(all_set_sizes) == 1:
    axes = axes[None]

for j, set_size in enumerate(all_set_sizes):

    full_mean_resultant_length_sweep = utils.two_dim_sweep(original_alphas[set_size], original_gammas[set_size], step_along_mean_resultant_length_contour)
    selected_mean_resultant_length_sweep = utils.select_grid_of_parameters(**full_mean_resultant_length_sweep, num_param_set=data_gen_args.num_contour_params)
    full_kurtosis_sweep = utils.two_dim_sweep(original_alphas[set_size], original_gammas[set_size], step_along_kurtosis_contour)
    selected_kurtosis_sweep = utils.select_grid_of_parameters(**full_kurtosis_sweep, num_param_set=data_gen_args.num_contour_params)

    augmentation_alphas[set_size] = np.concatenate([np.array([original_alphas[set_size]]), selected_mean_resultant_length_sweep['alpha'], selected_kurtosis_sweep['alpha']])
    augmentation_gammas[set_size] = np.concatenate([np.array([original_gammas[set_size]]), selected_mean_resultant_length_sweep['gamma'], selected_kurtosis_sweep['gamma']])

    sweep_emission_first_resultant_vector_lengths[set_size] = np.concatenate([np.array([original_emission_first_resultant_vector_lengths[set_size]]), selected_mean_resultant_length_sweep['rho'], selected_kurtosis_sweep['rho']])
    sweep_emission_circular_kurtosis[set_size] = np.concatenate([np.array([original_emission_circular_kurtosis[set_size]]), selected_mean_resultant_length_sweep['k'], selected_kurtosis_sweep['k']])

    if data_gen_args.generation_source is None:
        axes[j,0].set_xlabel('alpha')
        axes[j,0].set_ylabel('gamma')

        vis_alphas = np.linspace(0.2, 1.0, data_gen_args.num_contour_params)

        axes[j,0].plot(full_mean_resultant_length_sweep['all_alphas'], full_mean_resultant_length_sweep['all_gammas'], color = 'blue', label = 'Fixed rho sweep')
        axes[j,0].plot(full_kurtosis_sweep['all_alphas'], full_kurtosis_sweep['all_gammas'], color = 'green', label = 'Fixed kurtosis sweep')
        axes[j,0].scatter(selected_mean_resultant_length_sweep['alpha'], selected_mean_resultant_length_sweep['gamma'], color = 'blue', marker = 'x', alpha=vis_alphas)
        axes[j,0].scatter(selected_kurtosis_sweep['alpha'], selected_kurtosis_sweep['gamma'], color = 'green', marker = 'x', alpha=vis_alphas)

        axes[j,0].plot([0.0, 2.0], [0.0, 0.0], color = 'black')
        y_lim = axes[j,0].get_ylim()[1]
        axes[j,0].plot([0.0, 0.0], [0.0, y_lim], color = 'black')
        axes[j,0].plot([2.0, 2.0], [0.0, y_lim], color = 'black')

        axes[j,1].set_xlabel('alpha')
        axes[j,2].set_xlabel('alpha')
        axes[j,1].set_ylabel('rho')
        axes[j,2].set_ylabel('kurtosis')

        axes[j,1].plot(full_mean_resultant_length_sweep['all_alphas'], full_mean_resultant_length_sweep['all_rhos'], color = 'blue')
        axes[j,1].plot(full_kurtosis_sweep['all_alphas'], full_kurtosis_sweep['all_rhos'], color = 'green')
        axes[j,1].scatter(selected_mean_resultant_length_sweep['alpha'], selected_mean_resultant_length_sweep['rho'], color = 'blue', marker = 'x', alpha=vis_alphas)
        axes[j,1].scatter(selected_kurtosis_sweep['alpha'], selected_kurtosis_sweep['rho'], color = 'green', marker = 'x', alpha=vis_alphas)

        axes[j,2].plot(full_mean_resultant_length_sweep['all_alphas'], full_mean_resultant_length_sweep['all_kurtosis'], color = 'blue')
        axes[j,2].plot(full_kurtosis_sweep['all_alphas'], full_kurtosis_sweep['all_kurtosis'], color = 'green')
        axes[j,2].scatter(selected_mean_resultant_length_sweep['alpha'], selected_mean_resultant_length_sweep['k'], color = 'blue', marker = 'x', alpha=vis_alphas)
        axes[j,2].scatter(selected_kurtosis_sweep['alpha'], selected_kurtosis_sweep['k'], color = 'green', marker = 'x', alpha=vis_alphas)

        theta_axis = torch.linspace(-torch.pi, +torch.pi, 100)

        pdf = symmetric_zero_mean_wrapped_stable(theta_axis, torch.tensor(original_alphas[set_size]), torch.tensor(original_gammas[set_size]), p_cut_off = generative_model.error_emissions.p_cut_off)
        assert pdf.min() > 0.0
        axes[j,3].plot(theta_axis.numpy(), pdf.numpy(), color = 'black')
        axes[j,4].plot(theta_axis.numpy(), pdf.numpy(), color = 'black')

        for va, al, ga in zip(vis_alphas, selected_mean_resultant_length_sweep['alpha'], selected_mean_resultant_length_sweep['gamma']):
            pdf = symmetric_zero_mean_wrapped_stable(theta_axis, torch.tensor(al), torch.tensor(ga), p_cut_off = generative_model.error_emissions.p_cut_off)
            assert pdf.min() > 0.0
            axes[j,3].plot(theta_axis.numpy(), pdf.numpy(), color = 'blue', alpha = va)

        for va, al, ga in zip(vis_alphas, selected_kurtosis_sweep['alpha'], selected_kurtosis_sweep['gamma']):
            pdf = symmetric_zero_mean_wrapped_stable(theta_axis, torch.tensor(al), torch.tensor(ga), p_cut_off = generative_model.error_emissions.p_cut_off)
            assert pdf.min() > 0.0
            axes[j,4].plot(theta_axis.numpy(), pdf.numpy(), color = 'green', alpha = va)

        axes[j,0].legend()


if data_gen_args.generation_source is None:
    fig.savefig(os.path.join(resume_path, f'circular_statistics_sweep_{data_gen_args.synthetic_data_code}.png'))

dataset_sizes = {N: dg.all_deltas.shape[0] for N, dg in dataset_generator.data_generators.items()}
num_reps = data_gen_args.num_synthetic_generation_repeats_per_emission_distribution
num_params = 1 + 2 * data_gen_args.num_contour_params


if data_gen_args.generation_source is None:

    num_generating_params = num_params
    num_infering_params = num_params

    if args.swap_type != 'spike_and_slab':
        all_mean_function_eval = {N: np.zeros([num_reps, num_generating_params, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}
    else:
        all_mean_function_eval = None
    all_synthetic_prior_vectors = {N: np.zeros([num_reps, num_generating_params, dataset_sizes[N], N + 1]).astype(float) for N in all_set_sizes}
    all_synthetic_components = {N: np.zeros([num_reps, num_generating_params, dataset_sizes[N]]) for N in all_set_sizes}
    all_synthetic_estimate_errors = {N: np.zeros([num_reps, num_generating_params, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}

else:
    loaded_data = np.load(data_gen_args.generation_source, allow_pickle=True).item()
    loaded_errors = loaded_data['generated_data']['errors']
    
    num_generating_params = list(loaded_errors.values())[0].shape[1]
    num_infering_params = num_params

    assert all([num_reps == loaded_errors[N].shape[0] for N in loaded_errors.keys()])
    assert all([num_generating_params == loaded_errors[N].shape[1] for N in loaded_errors.keys()])



all_reinferred_total_log_likelihoods = {N: np.zeros([num_reps, num_generating_params, num_infering_params, dataset_sizes[N]]).astype(float) for N in all_set_sizes}
all_inmodel_total_log_likelihoods = {N: np.zeros([num_reps, num_generating_params, num_infering_params, dataset_sizes[N]]).astype(float) for N in all_set_sizes}

all_reinferred_particle_non_uniform_weights = {N: np.zeros([num_reps, num_generating_params, num_infering_params, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}
all_reinferred_particle_uniform_weights = {N: np.zeros([num_reps, num_generating_params, num_infering_params, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}
all_reinferred_particle_mean_resultant_lengths = {N: np.zeros([num_reps, num_generating_params, num_infering_params]).astype(float) for N in all_set_sizes}
all_reinferred_particle_circular_kurtosis = {N: np.zeros([num_reps, num_generating_params, num_infering_params]).astype(float) for N in all_set_sizes}
all_inmodel_particle_non_uniform_weights = {N: np.zeros([num_reps, num_generating_params, num_infering_params, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}
all_inmodel_particle_uniform_weights = {N: np.zeros([num_reps, num_generating_params, num_infering_params, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}
all_inmodel_particle_mean_resultant_lengths = {N: np.zeros([num_reps, num_generating_params, num_infering_params]).astype(float) for N in all_set_sizes}
all_inmodel_particle_circular_kurtosis = {N: np.zeros([num_reps, num_generating_params, num_infering_params]).astype(float) for N in all_set_sizes}


emission_first_resultant_vector_length = {N: [wrapped_stable_mean_resultant_length(a, g) for a, g in zip(augmentation_alphas[N], augmentation_gammas[N])] for N in all_set_sizes}
emission_circular_kurtosis = {N: [wrapped_stable_kurtosis(a, g) for a, g in zip(augmentation_alphas[N], augmentation_gammas[N])] for N in all_set_sizes}


with torch.no_grad():
    
    for i, (set_size, dg) in enumerate(dataset_generator.data_generators.items()):
        
        for rep_idx in tqdm(range(num_reps)):

            print(f'Starting repeat {rep_idx + 1} of {num_reps} for N={set_size}. Iterator = dataset generated from fresh params')

            if not args.shared_swap_function:
                variational_model = variational_models[set_size]

            all_deltas: _T = dg.all_deltas
            component_estimate_deltas = all_deltas[...,1]
            all_relevant_deltas = all_deltas[...,delta_dimensions]
        
            for generating_param_idx in tqdm(range(num_generating_params)):

                if data_gen_args.generation_source is None:

                    generating_alpha = augmentation_alphas[set_size][generating_param_idx]
                    generating_gamma = augmentation_gammas[set_size][generating_param_idx]
                    kwargs_for_sample_from_components = {"alpha": torch.tensor(generating_alpha), "gamma": torch.tensor(generating_gamma)}

                    if args.swap_type == 'spike_and_slab':
                        kwargs_for_generate_pi_vectors = dict(batch_size = all_deltas.shape[0])
                        new_synth_data_batch = generative_model.full_data_generation(
                            set_size = set_size, vm_means = component_estimate_deltas, 
                            kwargs_for_generate_pi_vectors = kwargs_for_generate_pi_vectors, kwargs_for_sample_from_components=kwargs_for_sample_from_components
                        )

                    else:
                        feature_mu = inference_mean_only(generative_model=generative_model, variational_model=variational_model, deltas=all_relevant_deltas)
                        f_mean = variational_model.reparameterised_sample(
                            num_samples = 1, mu = feature_mu, 
                            sigma_chol = torch.zeros(feature_mu.shape[0], feature_mu.shape[0], dtype=feature_mu.dtype, device=feature_mu.device),         # i.e. just the mean!
                            M = all_relevant_deltas.shape[0], N = all_relevant_deltas.shape[1]
                        )                                                           # [batch, N]

                        kwargs_for_generate_pi_vectors = dict(model_evaulations = f_mean, make_spike_and_slab=data_gen_args.make_spike_and_slab)
                        new_synth_data_batch: Dict[str, _T] = generative_model.full_data_generation(
                            set_size = set_size, vm_means = component_estimate_deltas, 
                            kwargs_for_generate_pi_vectors = kwargs_for_generate_pi_vectors, kwargs_for_sample_from_components=kwargs_for_sample_from_components
                        )
                        new_mean_function_eval = f_mean.squeeze(0)

                        all_mean_function_eval[set_size][rep_idx,generating_param_idx] = new_mean_function_eval.cpu().numpy()

                    synthetic_errors: _T = rectify_angles(new_synth_data_batch['samples'].squeeze(0).unsqueeze(-1) - component_estimate_deltas)
                    synthetic_errors_2: _T = rectify_angles(new_synth_data_batch['samples'].squeeze(0).unsqueeze(-1) + dg.all_target_zetas[:,0] - dg.all_target_zetas.squeeze(-1))
                    assert torch.isclose(synthetic_errors, synthetic_errors_2).all()
                    
                    all_synthetic_prior_vectors[set_size][rep_idx,generating_param_idx] = new_synth_data_batch['pi_vectors'].squeeze(0).cpu().numpy()
                    all_synthetic_components[set_size][rep_idx,generating_param_idx] = new_synth_data_batch['betas'].squeeze(0).cpu().numpy()
                    all_synthetic_estimate_errors[set_size][rep_idx,generating_param_idx] = synthetic_errors.cpu().numpy()

                else:
                    synthetic_errors = torch.tensor(loaded_errors[set_size][rep_idx,generating_param_idx]).cuda()

                for inferring_param_idx, (inferring_alpha, inferring_gamma) in enumerate(zip(augmentation_alphas[set_size], augmentation_gammas[set_size])):

                    kwargs_for_inference = {"alpha": torch.tensor(inferring_alpha), "gamma": torch.tensor(inferring_gamma)}

                    with torch.no_grad():
                        if args.swap_type == 'spike_and_slab':
                            reinferred_elbo_terms = get_elbo_terms_spike_and_slab(
                                generative_model, synthetic_errors, synthetic_errors.shape[0], synthetic_errors.shape[1], 'error',
                                kwargs_for_individual_component_likelihoods = kwargs_for_inference
                            )
                        else:
                            reinferred_elbo_terms = get_elbo_terms(
                                variational_model, generative_model, all_relevant_deltas, synthetic_errors, args.monte_carlo_gp_samples, 'error', False,
                                kwargs_for_individual_component_likelihoods = kwargs_for_inference
                            )
                        new_reinferred_total_log_likelihoods = reinferred_elbo_terms['unaggregated_lh'].sum(-1).mean(0).log().cpu().numpy() # [num trials]
                        assert np.isclose(reinferred_elbo_terms['llh_term'].item(), new_reinferred_total_log_likelihoods.sum())
                        
                        reinferred_residual_estimation_weights = generative_model.empirical_residual_distribution_weights(
                            reinferred_elbo_terms['posterior'], synthetic_errors, kwargs_for_individual_component_likelihoods = kwargs_for_inference
                        )
                        new_mrv_reinferred = mean_resultant_length_from_angles(synthetic_errors, reinferred_residual_estimation_weights['particle_weights_total'])
                        new_ckr_reinferred = kurtosis_from_angles(synthetic_errors, reinferred_residual_estimation_weights['particle_weights_total'])

                        all_reinferred_total_log_likelihoods[set_size][rep_idx,generating_param_idx,inferring_param_idx] = new_reinferred_total_log_likelihoods
                        all_reinferred_particle_uniform_weights[set_size][rep_idx,generating_param_idx,inferring_param_idx] = reinferred_residual_estimation_weights['particle_weights_uniform'].cpu().numpy()
                        all_reinferred_particle_non_uniform_weights[set_size][rep_idx,generating_param_idx,inferring_param_idx] = reinferred_residual_estimation_weights['particle_weights_non_uniform'].cpu().numpy()
                        all_reinferred_particle_mean_resultant_lengths[set_size][rep_idx,generating_param_idx,inferring_param_idx] = new_mrv_reinferred
                        all_reinferred_particle_circular_kurtosis[set_size][rep_idx,generating_param_idx,inferring_param_idx] = new_ckr_reinferred

                        if data_gen_args.generation_source is None:
                            
                            total_inmodel_log_likelihood, inmodel_posterior, inmodel_pdf_grid = generative_model.get_marginalised_log_likelihood(
                                estimation_deviations = synthetic_errors, pi_vectors = new_synth_data_batch['pi_vectors'],
                                kwargs_for_individual_component_likelihoods = kwargs_for_inference
                            )
                            inmodel_residual_estimation_weights = generative_model.empirical_residual_distribution_weights(
                                inmodel_posterior, synthetic_errors, kwargs_for_individual_component_likelihoods = kwargs_for_inference
                            )
                            new_mrv_inmodel = mean_resultant_length_from_angles(synthetic_errors, inmodel_residual_estimation_weights['particle_weights_total'])
                            new_ckr_inmodel = kurtosis_from_angles(synthetic_errors, inmodel_residual_estimation_weights['particle_weights_total'])

                            new_inmodel_total_log_likelihoods = inmodel_pdf_grid.sum(-1).mean(0).log().cpu().numpy() # [num trials]
                            assert np.isclose(total_inmodel_log_likelihood.item(), new_inmodel_total_log_likelihoods.sum())

                            all_inmodel_total_log_likelihoods[set_size][rep_idx,generating_param_idx,inferring_param_idx] = new_inmodel_total_log_likelihoods
                            all_inmodel_particle_uniform_weights[set_size][rep_idx,generating_param_idx,inferring_param_idx] = inmodel_residual_estimation_weights['particle_weights_uniform'].cpu().numpy()
                            all_inmodel_particle_non_uniform_weights[set_size][rep_idx,generating_param_idx,inferring_param_idx] = inmodel_residual_estimation_weights['particle_weights_non_uniform'].cpu().numpy()
                            all_inmodel_particle_mean_resultant_lengths[set_size][rep_idx,generating_param_idx,inferring_param_idx] = new_mrv_inmodel
                            all_inmodel_particle_circular_kurtosis[set_size][rep_idx,generating_param_idx,inferring_param_idx] = new_ckr_inmodel



if data_gen_args.generation_source is None:
    generated_data = {
            "function_eval": all_mean_function_eval,
        "component_priors": all_synthetic_prior_vectors,
        "components": all_synthetic_components,
        "errors": all_synthetic_estimate_errors,
    }

    residual_distribution_inmodel = {
        "particle_non_uniforn_weights": all_inmodel_particle_non_uniform_weights,
        "particle_uniform_weights": all_inmodel_particle_uniform_weights,
        "particle_mean_first_resultant_vector_lengths": all_inmodel_particle_mean_resultant_lengths,
        "particle_circular_kurtosis": all_inmodel_particle_circular_kurtosis,
    }

    original_model_loglikelihood_inmodel = all_inmodel_total_log_likelihoods

else:
    generated_data = data_gen_args.generation_source
    residual_distribution_inmodel = None
    original_model_loglikelihood_inmodel = None


np.save(
    data_destination,
    {
        'generated_data': generated_data,
        "original_model_loglikelihood": {
            "reinferred": all_reinferred_total_log_likelihoods,  # overall llh - get from unaggreated pdf_grid - [num_repeats, num trials]
            "inmodel": original_model_loglikelihood_inmodel  # overall llh - get from unaggreated pdf_grid - [num_repeats, num trials]
        },
        "residual_distribution_reinferred": {
            "particle_non_uniforn_weights": all_reinferred_particle_non_uniform_weights,
            "particle_uniform_weights": all_reinferred_particle_uniform_weights,
            "particle_mean_first_resultant_vector_lengths": all_reinferred_particle_mean_resultant_lengths,
            "particle_circular_kurtosis": all_reinferred_particle_circular_kurtosis,
        },
        "residual_distribution_inmodel": residual_distribution_inmodel,

        # This is all purely from emissions distribution, nothing to do with particle estimation...
        # If data_gen_args.generation_source is None, then this is both for generation and inference.
        # If data_gen_args.generation_source is not None, then this is just for inference
        # First of all of these is the original model info!
        "emissions_parameter_sweep_info": { 
            # "original_alpha": original_alphas,
            # "original_gamma": original_gammas,
            # "original_mean_first_resultant_vector_length": original_emission_first_resultant_vector_lengths,
            # "original_circular_kurtosis": original_emission_circular_kurtosis,
            "sweep_alphas": augmentation_alphas,
            "sweep_gammas": augmentation_gammas,
            "sweep_mean_first_resultant_vector_length": sweep_emission_first_resultant_vector_lengths,
            "sweep_circular_kurtosis": sweep_emission_circular_kurtosis,
        }
    }
)

args.write_to_yaml(config_yaml_destination)
