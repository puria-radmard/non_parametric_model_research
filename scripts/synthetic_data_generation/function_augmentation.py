# Load up a model and its training data, then go over and generate data from the (possibly augmented) mean f

import numpy as np
import os, torch, argparse

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

from purias_utils.error_modelling_torus.non_parametric_error_model.util import inference_mean_only


parser = argparse.ArgumentParser()
parser.add_argument('--resume_path', type = str)
parser.add_argument('--synthetic_data_code', type = str, required = True)
parser.add_argument('--mean_function_scale', type = float, required = False, default = 1.0)
parser.add_argument('--mean_function_offset', type = float, required = False, default = 0.0)
parser.add_argument('--other_augmentations', type = str, required = False, default = None, nargs = '+')
parser.add_argument('--num_synthetic_generation_repeats', type = int, required = True)
data_gen_args = parser.parse_args()

resume_path = data_gen_args.resume_path
args = ConfigNamepace.from_yaml_path(os.path.join(resume_path, "args.yaml"))
args.dict.pop('resume_path')

dataset_generator = load_experimental_data(args.dataset_name, args.train_indices_seed, args.train_indices_path, args.M_batch, args.M_test_per_set_size, args.num_models, args)
all_set_sizes = list(dataset_generator.data_generators.keys())

generative_model, variational_models, variational_model, D, delta_dimensions = setup_model_whole(
    **args.dict, all_set_sizes=all_set_sizes, trainable_kernel_delta=False, min_seps=None, resume_path=resume_path
)


data_destination = os.path.join(resume_path, f'synthetic_data_{data_gen_args.synthetic_data_code}.npy')
figure_destination = os.path.join(resume_path, f'synthetic_data_f_mean_evaluated_for_{data_gen_args.synthetic_data_code}.png')
config_yaml_destination = os.path.join(resume_path, f'synthetic_data_args_{data_gen_args.synthetic_data_code}.yaml')

assert resume_path is not None


for dest in [data_destination, figure_destination, config_yaml_destination]:
    if os.path.exists(dest):
        raise Exception('Cannot overwrite data! ' + dest)

assert data_gen_args.mean_function_scale == 1.0, "Need to reimplement mean_function_scale for fix_non_swap models"
assert data_gen_args.mean_function_offset == 0.0, "Need to reimplement mean_function_offset for fix_non_swap models"
make_spike_and_slab = False
if data_gen_args.other_augmentations is None:
    augmentation_function = lambda f, d: f
else:
    assert len(data_gen_args.other_augmentations) == 1, "Stacking augmentations needs to be reimplemented!"
    if data_gen_args.other_augmentations[0] == 'make_spike_and_slab':
        make_spike_and_slab = True
        assert args.swap_type != 'spike_and_slab'
    else:
        assert D == 1
        augmentation_function = getattr(utils, data_gen_args.other_augmentations[0])


dataset_sizes = {N: dg.all_deltas.shape[0] for N, dg in dataset_generator.data_generators.items()}
num_reps = data_gen_args.num_synthetic_generation_repeats


assert args.num_models == 1, "function_augmentation.py only supports Q=1, and num_synthetic_generation_repeats (K) will take the place of Q in the later stages!"


# Adding extra dimension to allow direct compatibility with sweep_circular_statistics
all_mean_function_eval = {N: np.zeros([num_reps, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}
all_synthetic_prior_vectors = {N: np.zeros([num_reps, dataset_sizes[N], N + 1]).astype(float) for N in all_set_sizes}
all_synthetic_components = {N: np.zeros([num_reps, dataset_sizes[N]]) for N in all_set_sizes}
all_synthetic_estimate_errors = {N: np.zeros([num_reps, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}

all_reinferred_total_log_likelihoods = {N: np.zeros([num_reps, dataset_sizes[N]]).astype(float) for N in all_set_sizes}
all_inmodel_total_log_likelihoods = {N: np.zeros([num_reps, dataset_sizes[N]]).astype(float) for N in all_set_sizes}

all_reinferred_particle_non_uniform_weights = {N: np.zeros([num_reps, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}
all_reinferred_particle_uniform_weights = {N: np.zeros([num_reps, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}
all_reinferred_particle_mean_resultant_lengths = {N: np.zeros([num_reps]).astype(float) for N in all_set_sizes}
all_reinferred_particle_circular_kurtosis = {N: np.zeros([num_reps]).astype(float) for N in all_set_sizes}

all_inmodel_particle_non_uniform_weights = {N: np.zeros([num_reps, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}
all_inmodel_particle_uniform_weights = {N: np.zeros([num_reps, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}
all_inmodel_particle_mean_resultant_lengths = {N: np.zeros([num_reps]).astype(float) for N in all_set_sizes}
all_inmodel_particle_circular_kurtosis = {N: np.zeros([num_reps]).astype(float) for N in all_set_sizes}


with torch.no_grad():

    for rep_idx in range(num_reps):

        for i, (set_size, dg) in enumerate(dataset_generator.data_generators.items()):
            
            if not args.shared_swap_function:
                variational_model = variational_models[set_size]

            all_deltas: _T = dg.all_deltas
            component_estimate_deltas = all_deltas[...,1]
            all_relevant_deltas = all_deltas[None,...,delta_dimensions] # Q = 1

            if args.swap_type == 'spike_and_slab':
                assert data_gen_args.mean_function_scale == 1.0, "Cannot have non-identity data_gen_args.mean_function_scale for spike and slab swap model"
                assert data_gen_args.mean_function_offset == 0.0, "Cannot have non-zero data_gen_args.mean_function_offset for spike and slab swap model"
                assert data_gen_args.other_augmentations == None, "Cannot freely augment spike and slab swap model confusion function"

                new_synth_data_batch = generative_model.full_data_generation(set_size = set_size, vm_means = component_estimate_deltas, kwargs_for_generate_pi_vectors = {'batch_size': component_estimate_deltas.shape[1]})
                new_mean_function_eval = new_synth_data_batch['exp_grid'].squeeze(0).log()

            else:
                feature_mu = inference_mean_only(generative_model=generative_model, variational_model=variational_model, deltas=all_relevant_deltas)    # [Q (1), 4800]
                f_mean = variational_model.reparameterised_sample(
                    num_samples = 1, mu = feature_mu, 
                    sigma_chol = torch.zeros(feature_mu.shape[0], feature_mu.shape[1], feature_mu.shape[1], dtype=feature_mu.dtype, device=feature_mu.device),         # i.e. just the mean!
                    M = all_relevant_deltas.shape[1], N = all_relevant_deltas.shape[2]
                )                                                           # [1, 1, batch, N]

                augmented_f_mean = augmentation_function(f_mean.squeeze(0), all_relevant_deltas.squeeze(-1)).unsqueeze(0)

                kwargs_for_generate_pi_vectors = dict(model_evaulations = augmented_f_mean, make_spike_and_slab = make_spike_and_slab)
                new_synth_data_batch: Dict[str, _T] = generative_model.full_data_generation(set_size = set_size, vm_means = component_estimate_deltas, kwargs_for_generate_pi_vectors = kwargs_for_generate_pi_vectors)
                new_mean_function_eval = augmented_f_mean.squeeze(0)

            synthetic_errors: _T = rectify_angles(new_synth_data_batch['samples'].squeeze(0).unsqueeze(-1) - component_estimate_deltas)
            synthetic_errors_2: _T = rectify_angles(new_synth_data_batch['samples'].squeeze(0).unsqueeze(-1) + dg.all_target_zetas[:,0] - dg.all_target_zetas.squeeze(-1))
            assert torch.isclose(synthetic_errors, synthetic_errors_2).all()

            with torch.no_grad():
                if args.swap_type == 'spike_and_slab':
                    reinferred_elbo_terms = get_elbo_terms_spike_and_slab(
                        generative_model, synthetic_errors.unsqueeze(0), synthetic_errors.shape[0], synthetic_errors.shape[1], 'error'  # assuming original Q = 1 again!!!
                    )
                else:
                    reinferred_elbo_terms = get_elbo_terms(
                        variational_model, generative_model, all_relevant_deltas, synthetic_errors.unsqueeze(0), args.monte_carlo_gp_samples, 'error', False
                    )
                new_reinferred_log_likelihoods_per_data_point = reinferred_elbo_terms['likelihood_per_datapoint'] # [1, num trials]
                assert np.isclose(reinferred_elbo_terms['total_log_likelihood'].item(), new_reinferred_log_likelihoods_per_data_point.sum().item())

                print('NB: directional states done with assumption of Q=1 in generating model!')
                reinferred_residual_estimation_weights = generative_model.empirical_residual_distribution_weights(reinferred_elbo_terms['posterior'], synthetic_errors.unsqueeze(0))
                new_mrv_reinferred = mean_resultant_length_from_angles(synthetic_errors, reinferred_residual_estimation_weights['particle_weights_total'])      # Again assume Q = 1!
                new_ckr_reinferred = kurtosis_from_angles(synthetic_errors, reinferred_residual_estimation_weights['particle_weights_total'])       # Again assume Q = 1!

                total_inmodel_log_likelihood, new_inmodel_log_likelihoods_per_data_point, inmodel_posterior = generative_model.get_marginalised_log_likelihood(
                    estimation_deviations = synthetic_errors.unsqueeze(0), pi_vectors = new_synth_data_batch['pi_vectors'],
                )
                inmodel_residual_estimation_weights = generative_model.empirical_residual_distribution_weights(inmodel_posterior, synthetic_errors.unsqueeze(0))
                new_mrv_inmodel = mean_resultant_length_from_angles(synthetic_errors, inmodel_residual_estimation_weights['particle_weights_total'])        # Again assume Q = 1!
                new_ckr_inmodel = kurtosis_from_angles(synthetic_errors, inmodel_residual_estimation_weights['particle_weights_total'])     # Again assume Q = 1!

                assert np.isclose(total_inmodel_log_likelihood.item(), new_inmodel_log_likelihoods_per_data_point[-1].sum().item())

            all_mean_function_eval[set_size][rep_idx] = all_mean_function_eval[set_size][rep_idx] * np.nan if args.swap_type == 'spike_and_slab' else new_mean_function_eval.cpu().numpy()
            all_synthetic_prior_vectors[set_size][rep_idx] = new_synth_data_batch['pi_vectors'].squeeze(0).cpu().numpy()
            all_synthetic_components[set_size][rep_idx] = new_synth_data_batch['betas'].squeeze(0).cpu().numpy()
            all_synthetic_estimate_errors[set_size][rep_idx] = synthetic_errors.cpu().numpy()

            all_reinferred_total_log_likelihoods[set_size][rep_idx] = new_reinferred_log_likelihoods_per_data_point.cpu().numpy()
            all_inmodel_total_log_likelihoods[set_size][rep_idx] = new_inmodel_log_likelihoods_per_data_point.cpu().numpy()

            all_reinferred_particle_uniform_weights[set_size][rep_idx] = reinferred_residual_estimation_weights['particle_weights_uniform'].cpu().numpy()
            all_reinferred_particle_non_uniform_weights[set_size][rep_idx] = reinferred_residual_estimation_weights['particle_weights_non_uniform'].cpu().numpy()
            all_reinferred_particle_mean_resultant_lengths[set_size][rep_idx] = new_mrv_reinferred
            all_reinferred_particle_circular_kurtosis[set_size][rep_idx] = new_ckr_reinferred

            all_inmodel_particle_uniform_weights[set_size][rep_idx] = inmodel_residual_estimation_weights['particle_weights_uniform'].cpu().numpy()
            all_inmodel_particle_non_uniform_weights[set_size][rep_idx] = inmodel_residual_estimation_weights['particle_weights_non_uniform'].cpu().numpy()
            all_inmodel_particle_mean_resultant_lengths[set_size][rep_idx] = new_mrv_inmodel
            all_inmodel_particle_circular_kurtosis[set_size][rep_idx] = new_ckr_inmodel

                

np.save(
    data_destination,
    {
        'generated_data': {
            "function_eval_mean": all_mean_function_eval,
            "component_priors": all_synthetic_prior_vectors,
            "components": all_synthetic_components,
            "errors": all_synthetic_estimate_errors,
        },
        "loglikelihood": {
            "reinferred": all_reinferred_total_log_likelihoods,  # overall llh - get from unaggreated pdf_grid - [num_repeats, num trials]
            "inmodel": all_inmodel_total_log_likelihoods  # overall llh - get from unaggreated pdf_grid - [num_repeats, num trials]
        },
        "residual_distribution_reinferred": {
            "particle_non_uniforn_weights": all_reinferred_particle_non_uniform_weights,
            "particle_uniform_weights": all_reinferred_particle_uniform_weights,
            "particle_mean_first_resultant_vector_lengths": all_reinferred_particle_mean_resultant_lengths,
            "particle_circular_kurtosis": all_reinferred_particle_circular_kurtosis
        },
        "residual_distribution_inmodel": {
            "particle_non_uniforn_weights": all_inmodel_particle_non_uniform_weights,
            "particle_uniform_weights": all_inmodel_particle_uniform_weights,
            "particle_mean_first_resultant_vector_lengths": all_inmodel_particle_mean_resultant_lengths,
            "particle_circular_kurtosis": all_inmodel_particle_circular_kurtosis
        },
    }
)


args.write_to_yaml(config_yaml_destination)
