import torch
from torch.distributions.von_mises import _log_modified_bessel_fn
from purias_utils.maths.circular_statistics import kurtosis_from_angles, wrapped_stable_kurtosis, fit_symmetric_zero_mean_wrapped_stable_to_samples

from matplotlib import pyplot as plt

from purias_utils.error_modelling_torus.non_parametric_error_model.util import likelihood_inner, get_elbo_terms, get_elbo_terms_spike_and_slab


def get_emissions_kurtosis(N, emission_type, generative_model):
    
    if emission_type == 'von_mises':
        # WAIT THIS SHOULD BE ZERO ANYWAY!
        kappa = generative_model.error_emissions.emission_parameter(N)
        equivalent_gamma = (_log_modified_bessel_fn(kappa, 0) - _log_modified_bessel_fn(kappa, 1)).sqrt()
        emissions_kurtosis = wrapped_stable_kurtosis(2.0, equivalent_gamma.detach().cpu().numpy()).item()

    if emission_type == 'wrapped_stable':
        alpha, gamma = generative_model.error_emissions.emission_parameter(N)
        emissions_kurtosis = wrapped_stable_kurtosis(alpha.detach().cpu().numpy(), gamma.detach().cpu().numpy()).item()

    return emissions_kurtosis


def errors_kurtosis_values(estimates, num_iter = 300, num_fits=10, do_fitted_kurtosis=True):

    estimates_kurtosis_from_particles = kurtosis_from_angles(estimates.cpu().numpy(), 1.0)

    best_loss = torch.inf

    if do_fitted_kurtosis:
        for _ in range(num_fits):
            
            a0=torch.rand([1]) * 1.8 + 0.2   # a
            g0=torch.randn([1]).exp() * 2  # g

            all_losses, all_alphas, all_gammas, (fitted_alpha_raw, fitted_gamma_raw) = fit_symmetric_zero_mean_wrapped_stable_to_samples(
                alpha_0 = a0, gamma_0 = g0, samples = estimates, weights = 1.0, num_iter = num_iter, lr = 0.1
            )

            if all_losses[-1] < best_loss:
                fitted_alpha = fitted_alpha_raw.tanh() + 1.0
                fitted_gamma = fitted_gamma_raw.exp()
                best_fitted_kurtosis = wrapped_stable_kurtosis(fitted_alpha, fitted_gamma).item()

    else:
        best_fitted_kurtosis = torch.nan

    return best_fitted_kurtosis, estimates_kurtosis_from_particles



def residuals_kurtosis_values(residual_knot_locations, residual_knot_weights, num_iter = 300, num_fits=10, do_fitted_kurtosis=True):

    assert residual_knot_locations.shape == residual_knot_weights.shape
    residual_kurtosis_from_particles = kurtosis_from_angles(residual_knot_locations.cpu().numpy(), residual_knot_weights.cpu().numpy())

    best_loss = torch.inf

    if do_fitted_kurtosis:
        for _ in range(num_fits):
            
            a0=torch.rand([1]) * 1.8 + 0.2   # a
            g0=torch.randn([1]).exp() * 2  # g

            all_losses, all_alphas, all_gammas, (fitted_alpha_raw, fitted_gamma_raw) = fit_symmetric_zero_mean_wrapped_stable_to_samples(
                alpha_0 = a0, gamma_0 = g0, samples = residual_knot_locations, weights = residual_knot_weights, num_iter = num_iter, lr = 0.1
            )

            if all_losses[-1] < best_loss:
                fitted_alpha = fitted_alpha_raw.tanh() + 1.0
                fitted_gamma = fitted_gamma_raw.exp()
                best_fitted_kurtosis = wrapped_stable_kurtosis(fitted_alpha, fitted_gamma).item()
    
    else:
        best_fitted_kurtosis = torch.nan

    return best_fitted_kurtosis, residual_kurtosis_from_particles



def generate_full_residual_particle_set(generative_model, variational_model, swap_type, all_relevant_deltas, all_errors, num_gp_samples = 128):
    """
    particle_locations are locations of particles on [-\pi, \pi)
    
    particle_weights_non_uniform are weights due to the von Mises/wrapped stable components --> resolve to just being the posterior for that component

    particle_weights_uniform are weights due to the uniform distribution --> resolve to just being the emission pdf multiplied by the posterior for the uniform distribution
        NB: we are making the approximation that particle_locations densely populates the circle, which means this component needs not be evaluated on another grid...

    All items of shape [M*N], i.e. multiply by N
    sum of particle_weights_non_uniform and particle_weights_uniform will be M originally, so we renormalise so that the average value amongst them is 1, i.e. 
    """

    M, N, D = all_relevant_deltas.shape
    assert list(all_errors.shape) == [M, N]

    with torch.no_grad():
        if swap_type == 'spike_and_slab':
            _, component_posterior_vectors, _ = get_elbo_terms_spike_and_slab(
                generative_model, all_errors, M, N, 'error', True
            )
        else:
            _, _, component_posterior_vectors, _ = get_elbo_terms(
                variational_model, generative_model, all_relevant_deltas, all_errors,
                M, N, num_gp_samples, 'error', False, True
            )
    
        particle_locations = all_errors.flatten()
        particle_weights_non_uniform = component_posterior_vectors[:,1:].flatten()

        raise Exception('This approximation of the uniform weights is broken')
        particle_weights_uniform = (component_posterior_vectors[:,[0]] * generative_model.error_emissions.individual_component_likelihoods_from_estimate_deviations_inner(N, all_errors)).flatten()

        particle_weights_non_uniform = particle_weights_non_uniform * N
        particle_weights_uniform = particle_weights_uniform * N

    raise Exception('Dont use this one anymore, use results directly from synthetic data generation')

    return (
        particle_locations,
        particle_weights_non_uniform,
        particle_weights_uniform
    )

