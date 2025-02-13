import torch
import numpy as np
from torch import Tensor as _T

from purias_utils.maths.circular_statistics import kurtosis_from_angles, mean_resultant_length_from_angles, wrapped_stable_kurtosis, wrapped_stable_mean_resultant_length
from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles

from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model import NonParametricSwapErrorsGenerativeModel
from purias_utils.error_modelling_torus.non_parametric_error_model.variational_approx import NonParametricSwapErrorsVariationalModel
from purias_utils.error_modelling_torus.data_utils.base import EstimateDataLoaderBase


all_lobe_removal_functions = ['lobe_removal_linear_increase', 'lobe_removal_flatten_lobes', 'lobe_removal_remove_lobes']



def lobe_removal_flatten_lobes(f_mean, relevant_deltas):

    assert relevant_deltas.shape == f_mean.shape

    neg_mask = relevant_deltas < 0.0
    pos_mask = relevant_deltas > 0.0

    neg_peak = f_mean[neg_mask].max()
    neg_peak_index = f_mean[neg_mask].argmax()
    neg_peak_location = relevant_deltas[neg_mask][neg_peak_index]

    pos_peak = f_mean[pos_mask].max()
    pos_peak_index = f_mean[pos_mask].argmax()
    pos_peak_location = relevant_deltas[pos_mask][pos_peak_index]

    augmented_f_mean = f_mean.clone()
    augmented_f_mean[torch.logical_and(neg_mask, relevant_deltas > neg_peak_location)] = neg_peak
    augmented_f_mean[torch.logical_and(pos_mask, relevant_deltas < pos_peak_location)] = pos_peak

    return augmented_f_mean



def lobe_removal_remove_lobes(f_mean, relevant_deltas):

    assert relevant_deltas.shape == f_mean.shape

    neg_mask = relevant_deltas < 0.0
    pos_mask = relevant_deltas > 0.0

    neg_trough = f_mean[neg_mask].min()
    neg_peak_index = f_mean[neg_mask].argmax()
    neg_peak_location = relevant_deltas[neg_mask][neg_peak_index]

    pos_trough = f_mean[pos_mask].min()
    pos_peak_index = f_mean[pos_mask].argmax()
    pos_peak_location = relevant_deltas[pos_mask][pos_peak_index]

    augmented_f_mean = f_mean.clone()
    augmented_f_mean[torch.logical_and(neg_mask, relevant_deltas > neg_peak_location)] = neg_trough
    augmented_f_mean[torch.logical_and(pos_mask, relevant_deltas < pos_peak_location)] = pos_trough

    return augmented_f_mean


def lobe_removal_linear_increase(f_mean, relevant_deltas):

    assert relevant_deltas.shape == f_mean.shape

    neg_mask = relevant_deltas < 0.0
    pos_mask = relevant_deltas > 0.0

    neg_tail_index = relevant_deltas[neg_mask].argmin()
    neg_tail_location = relevant_deltas[neg_mask].min()
    neg_tail = f_mean[neg_mask][neg_tail_index]

    pos_tail_index = relevant_deltas[pos_mask].argmin()
    pos_tail_location = relevant_deltas[pos_mask].max()
    pos_tail = f_mean[pos_mask][pos_tail_index]

    neg_peak = f_mean[neg_mask].max()
    neg_peak_index = f_mean[neg_mask].argmax()
    neg_peak_location = relevant_deltas[neg_mask][neg_peak_index]

    pos_peak = f_mean[pos_mask].max()
    pos_peak_index = f_mean[pos_mask].argmax()
    pos_peak_location = relevant_deltas[pos_mask][pos_peak_index]

    neg_grad = ((neg_peak - neg_tail) / (neg_peak_location - neg_tail_location)).item()
    pos_grad = ((pos_peak - pos_tail) / (pos_peak_location - pos_tail_location)).item()

    augmented_f_mean = f_mean.clone()

    neg_replacement_mask = torch.logical_and(neg_mask, relevant_deltas > neg_peak_location)
    augmented_f_mean[neg_replacement_mask] = neg_peak + neg_grad * (relevant_deltas[neg_replacement_mask] - neg_peak_location)

    pos_replacement_mask = torch.logical_and(pos_mask, relevant_deltas < pos_peak_location)
    augmented_f_mean[pos_replacement_mask] = pos_peak + pos_grad * (relevant_deltas[pos_replacement_mask] - pos_peak_location)

    return augmented_f_mean



def two_dim_sweep(initial_alpha, initial_gamma, step_along_function, alpha_thres_upper = 1.99, alpha_thers_lower = 0.75, gamma_thres_lower = 0.0001, max_steps = 100000):
    
    alphas = [initial_alpha]
    gammas = [initial_gamma]
    rhos = [wrapped_stable_mean_resultant_length(initial_alpha, initial_gamma)]
    ks = [wrapped_stable_kurtosis(initial_alpha, initial_gamma)]

    # Increasing alpha search
    alpha, gamma = np.array([initial_alpha]), np.array([initial_gamma])
    for t in range(max_steps):
        alpha, gamma = step_along_function(alpha, gamma, increasing_alpha=True)
        if alpha>alpha_thres_upper or alpha<alpha_thers_lower or gamma<gamma_thres_lower:
            break
        alphas.append(alpha[0])
        gammas.append(gamma[0])
        rhos.append(wrapped_stable_mean_resultant_length(alpha, gamma)[0])
        ks.append(wrapped_stable_kurtosis(alpha, gamma)[0])

    # Decreasing alpha search
    alpha, gamma = np.array([initial_alpha]), np.array([initial_gamma])
    for t in range(max_steps):
        alpha, gamma = step_along_function(alpha, gamma, increasing_alpha=False)
        if alpha>alpha_thres_upper or alpha<alpha_thers_lower or gamma<gamma_thres_lower:
            break
        alphas.insert(0, alpha[0])
        gammas.insert(0, gamma[0])
        rhos.insert(0, wrapped_stable_mean_resultant_length(alpha, gamma)[0])
        ks.insert(0, wrapped_stable_kurtosis(alpha, gamma)[0])

    return {'all_alphas': np.array(alphas), 'all_gammas': np.array(gammas), 'all_rhos': np.array(rhos), 'all_kurtosis': np.array(ks)}


def select_grid_of_parameters(all_alphas, all_gammas, all_rhos, all_kurtosis, num_param_set: int = 10):
    close_alphas = np.linspace(all_alphas.min(), all_alphas.max(), num_param_set)
    selected_alphas, selected_gammas = [], []
    selected_rhos, selected_kurtosis = [], []
    for ca in close_alphas:
        idx = np.abs(all_alphas - ca).argmin()
        selected_alphas.append(all_alphas[idx])
        selected_gammas.append(all_gammas[idx])
        selected_rhos.append(all_rhos[idx])
        selected_kurtosis.append(all_kurtosis[idx])
    return {'alpha': np.array(selected_alphas), 'gamma': np.array(selected_gammas), 'rho': np.array(selected_rhos), 'k': np.array(selected_kurtosis)}



def generate_and_log_all_information_from_synthetic_data_generation(
        N: int, I: int, new_synth_data_batch: dict, data_generator: EstimateDataLoaderBase, 
        swap_type: str, all_relevant_deltas: _T, new_mean_function_eval: _T,
        estimation_dimension_deltas: _T, all_mean_function_eval: dict, 
        all_synthetic_prior_vectors: dict, all_synthetic_components: dict, 
        all_synthetic_estimate_errors: dict, all_reinferred_total_log_likelihoods: dict, all_inmodel_total_log_likelihoods: dict,
        all_reinferred_particle_uniform_weights: dict, all_reinferred_particle_non_uniform_weights: dict,
        all_reinferred_particle_mean_resultant_lengths: dict, all_reinferred_particle_circular_kurtosis: dict,
        all_inmodel_particle_uniform_weights: dict, all_inmodel_particle_non_uniform_weights: dict,
        all_inmodel_particle_mean_resultant_lengths: dict, all_inmodel_particle_circular_kurtosis: dict,
        generative_model: NonParametricSwapErrorsGenerativeModel, variational_model: NonParametricSwapErrorsVariationalModel
    ):

        synthetic_errors: _T = rectify_angles(new_synth_data_batch['samples'].squeeze(0).unsqueeze(-1) - estimation_dimension_deltas)
        synthetic_errors_2: _T = rectify_angles(new_synth_data_batch['samples'].squeeze(0).unsqueeze(-1) + data_generator.all_target_zetas[:,0] - data_generator.all_target_zetas.squeeze(-1))
        assert torch.isclose(synthetic_errors, synthetic_errors_2).all()

        with torch.no_grad():
            if swap_type == 'spike_and_slab':
                reinferred_elbo_terms = get_elbo_terms_spike_and_slab(
                    generative_model, synthetic_errors, synthetic_errors.shape[0], synthetic_errors.shape[1], 'error'
                )
            else:
                reinferred_elbo_terms = get_elbo_terms(
                    variational_model, generative_model, all_relevant_deltas, synthetic_errors, I, 'error', False
                )
            new_reinferred_total_log_likelihoods = reinferred_elbo_terms['unaggregated_lh'].sum(-1).mean(0).log().cpu().numpy() # [num trials]
            assert np.isclose(reinferred_elbo_terms['llh_term'].item(), new_reinferred_total_log_likelihoods.sum())
            
            reinferred_residual_estimation_weights = generative_model.empirical_residual_distribution_weights(reinferred_elbo_terms['posterior'], synthetic_errors)
            new_mrv_reinferred = mean_resultant_length_from_angles(synthetic_errors, reinferred_residual_estimation_weights['particle_weights_total'])
            new_ckr_reinferred = kurtosis_from_angles(synthetic_errors, reinferred_residual_estimation_weights['particle_weights_total'])

            total_inmodel_log_likelihood, inmodel_posterior, inmodel_pdf_grid = generative_model.get_marginalised_log_likelihood(
                estimation_deviations = synthetic_errors, pi_vectors = new_synth_data_batch['pi_vectors'],
            )
            inmodel_residual_estimation_weights = generative_model.empirical_residual_distribution_weights(inmodel_posterior, synthetic_errors)
            new_mrv_inmodel = mean_resultant_length_from_angles(synthetic_errors, inmodel_residual_estimation_weights['particle_weights_total'])
            new_ckr_inmodel = kurtosis_from_angles(synthetic_errors, inmodel_residual_estimation_weights['particle_weights_total'])

            new_inmodel_total_log_likelihoods = inmodel_pdf_grid.sum(-1).mean(0).log().cpu().numpy() # [num trials]
            assert np.isclose(total_inmodel_log_likelihood.item(), all_inmodel_total_log_likelihoods[N][-1].sum())

        all_mean_function_eval[N].append(new_mean_function_eval.cpu().numpy())
        all_synthetic_prior_vectors[N].append(new_synth_data_batch['pi_vectors'].squeeze(0).cpu().numpy())
        all_synthetic_components[N].append(new_synth_data_batch['betas'].squeeze(0).cpu().numpy())
        all_synthetic_estimate_errors[N].append(synthetic_errors.cpu().numpy())

        all_reinferred_total_log_likelihoods[N].append(new_reinferred_total_log_likelihoods) 
        all_inmodel_total_log_likelihoods[N].append(new_inmodel_total_log_likelihoods)

        all_reinferred_particle_uniform_weights[N].append(reinferred_residual_estimation_weights['particle_weights_uniform'].cpu().numpy())
        all_reinferred_particle_non_uniform_weights[N].append(reinferred_residual_estimation_weights['particle_weights_non_uniform'].cpu().numpy())
        all_reinferred_particle_mean_resultant_lengths[N].append(new_mrv_reinferred)
        all_reinferred_particle_circular_kurtosis[N].append(new_ckr_reinferred)

        all_inmodel_particle_uniform_weights[N].append(inmodel_residual_estimation_weights['particle_weights_uniform'].cpu().numpy())
        all_inmodel_particle_non_uniform_weights[N].append(inmodel_residual_estimation_weights['particle_weights_non_uniform'].cpu().numpy())
        all_inmodel_particle_mean_resultant_lengths[N].append(new_mrv_inmodel)
        all_inmodel_particle_circular_kurtosis[N].append(new_ckr_inmodel)
