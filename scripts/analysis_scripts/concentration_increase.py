# 11/9/2024
# Comparing the concentration increase (for the whole dataset) between raw p(error) and residual p(epsilon)

import torch
import numpy as np
from os.path import join
from tqdm import tqdm

import matplotlib.pyplot as plt

from purias_utils.maths.circular_statistics import kurtosis_from_angles, mean_resultant_length_from_angles, symmetric_zero_mean_wrapped_stable

from purias_utils.error_modelling_torus.data_utils.loading_utils import load_experimental_data

from purias_utils.util.arguments_yaml import ConfigNamepace

from non_parametric_model.scripts.analysis_scripts.utils import approximate_cdf



root_path = '/homes/pr450/repos/research_projects/error_modelling_torus/results_link/d_finalise_kurtosis_results_8_9_24/trained_on_data/mcmaster2022_e1_oricue_cue_AR2'
save_paths = [
    (join(root_path, "cue_dim_only_wrapped_stable/run_0"), "circular_statistics_sweep", True),                    # inference of BnP data with BnP
    (join(root_path, "spike_and_slab_wrapped_stable/run_0"), "circular_statistics_sweep", True),     # inference of SnS data with SnS
    (join(root_path, "cue_dim_only_wrapped_stable/run_0"), "inference_of_circular_statistics_sweep_from_mcmaster2022_e1_oricue_spike_and_slab_wrapped_stable", False),       # inference of SnS data with BnP
    (join(root_path, "spike_and_slab_wrapped_stable/run_0"), "inference_of_circular_statistics_sweep_from_mcmaster2022_e1_oricue_cue_dim_wrapped_stable", False),     # inference of BnP data with SnS
    (join(root_path, "cue_dim_only_wrapped_stable/run_0"), "circular_statistics_sweep_spike_and_slab", False),     # inference of flattened-BnP data with flattened-BnP
    (join(root_path, "cue_dim_only_wrapped_stable/run_0"), "inference_of_circular_statistics_sweep_with_spike_and_slab", False),     # inference of BnP data with flattened-BnP
    (join(root_path, "cue_dim_only_wrapped_stable/run_0"), "inference_of_circular_statistics_sweep_spike_and_slab_with_bnp", False),     # inference of flattened-BnP data with BnP
    (join(root_path, "cue_dim_only_wrapped_stable/run_0"), "inference_of_circular_statistics_sweep_spike_and_slab_with_bnp", False),     # inference of flattened-BnP data with BnP
]

theta_axis = torch.linspace(-torch.pi, +torch.pi, 10000)

cached_synthetic_errors = {}
cached_generating_sweep_alphas = {}
cached_generating_sweep_gammas = {}
cached_inference_distrubtion_synthetic_emissions_mrws = {}

all_synth_ks_stats_per_save_file = {}
all_real_ks_stats_per_save_file = {}

# Iterate over saves
# XXX: replace this with cross-inference option too
for (save_path, code_name, original_file_data_inference_flag) in save_paths:

    # Load up data and training information, including information about the residual particles on real data
    args = ConfigNamepace.from_yaml_path(join(save_path, "args.yaml"))
    dataset_generator = load_experimental_data(args.dataset_name, args.train_indices_seed, args.train_indices_path, args.M_batch, args.M_test_per_set_size, args)
    all_set_sizes = list(dataset_generator.data_generators.keys())
    recent_losses = np.load(join(save_path, "recent_losses.npy"), allow_pickle=True).item()
    bnp_mrvs = recent_losses['residual_distribution']['particle_mean_first_resultant_vector_lengths']
    bnp_ckrs = recent_losses['residual_distribution']['particle_circular_kurtosis']
    
    # Load up information about the synthetic data (residual and emission data included)
    data_path = join(save_path, f"synthetic_data_{code_name}.npy")
    print(data_path)
    synthetic_data_circular_statistics_sweep = np.load(data_path, allow_pickle=True).item()

    # This figure will be used for just one model at a time!
    fig, axes = plt.subplots(len(all_set_sizes), 4, figsize = (20, 5*len(all_set_sizes)))
    if len(all_set_sizes) == 1:
        axes = axes[None]
    raw_mrvs, raw_ckrs = {}, {}

    pdb_flag = True

    # Iterate over set sizes
    for iN, (set_size, dg) in enumerate(dataset_generator.data_generators.items()):
        
        ### Bays 2016 method
        # XXX: Actually - doesn't apply to lots of datasets because we have minimum margins between estimated features, so cannot assume uniform term like in the paper

        # Extract all information about the parameters **used for residual distribution inference**
        inference_sweep_alphas = torch.tensor(synthetic_data_circular_statistics_sweep['emissions_parameter_sweep_info']['sweep_alphas'][set_size])   # [generation_parameters]
        inference_sweep_gammas = torch.tensor(synthetic_data_circular_statistics_sweep['emissions_parameter_sweep_info']['sweep_gammas'][set_size])   # [generation_parameters]
        inference_emissions_p_evals = torch.stack([symmetric_zero_mean_wrapped_stable(theta_axis, a, g, 1000) for a, g in tqdm(zip(inference_sweep_alphas, inference_sweep_gammas))]).cpu().numpy() # [generation_parameters, 1000]
        integrals = (inference_emissions_p_evals * (theta_axis[1] - theta_axis[0]).cpu().numpy()).sum(-1)
        assert (np.abs(integrals - 1.0) < 0.001).all(), integrals
        inference_distrubtion_synthetic_emissions_mrws = synthetic_data_circular_statistics_sweep['emissions_parameter_sweep_info']['sweep_mean_first_resultant_vector_length'][set_size]

        # To make this work for cross-inferences, cache this info, and *make sure* save_paths is in the right order!
        if isinstance(synthetic_data_circular_statistics_sweep['generated_data'], str):
            cache_key = synthetic_data_circular_statistics_sweep['generated_data']  # i.e. path from elsewhere
            all_synthetic_errors = cached_synthetic_errors[cache_key]
            generating_sweep_alphas = cached_generating_sweep_alphas[cache_key]
            generating_sweep_gammas = cached_generating_sweep_gammas[cache_key]
            generative_distrubtion_synthetic_emissions_mrws = cached_inference_distrubtion_synthetic_emissions_mrws[cache_key]
        else:
            all_synthetic_errors = synthetic_data_circular_statistics_sweep['generated_data']['errors'][set_size]   # [repeats per parameter, generation_parameters = 2 * num synthetic parameters + 1, M, N]
            generating_sweep_alphas, generating_sweep_gammas = inference_sweep_alphas, inference_sweep_gammas
            generative_distrubtion_synthetic_emissions_mrws = inference_distrubtion_synthetic_emissions_mrws
            
            cache_key = data_path
            cached_synthetic_errors[cache_key] = all_synthetic_errors
            cached_generating_sweep_alphas[cache_key] = generating_sweep_alphas
            cached_generating_sweep_gammas[cache_key] = generating_sweep_gammas
            cached_inference_distrubtion_synthetic_emissions_mrws[cache_key] = generative_distrubtion_synthetic_emissions_mrws


        # Renormalise the particle weights for the residual distribution
        all_p_tilde_particle_weights = synthetic_data_circular_statistics_sweep['residual_distribution_reinferred']['particle_non_uniforn_weights'][set_size] + synthetic_data_circular_statistics_sweep['residual_distribution_reinferred']['particle_uniform_weights'][set_size]
        all_p_tilde_particle_weights = all_p_tilde_particle_weights / all_synthetic_errors.shape[2]


        # Actual emission distribution
        all_synthetic_particle_residual_mrws = synthetic_data_circular_statistics_sweep['residual_distribution_reinferred']['particle_mean_first_resultant_vector_lengths'][set_size]


        # Initialise logging of KS statistics
        num_repetitions, num_generating_params, num_infering_params, num_examples, N = all_p_tilde_particle_weights.shape
        all_synth_ks_stats = np.zeros([num_repetitions, num_generating_params, num_infering_params])

        ### Iterate over multiple things...
        assert N == set_size
        for generating_param_idx in tqdm(range(num_generating_params)):
            
            # Extract information that depends only on generation
            synthetic_emissions_mrw = generative_distrubtion_synthetic_emissions_mrws[generating_param_idx].item()
            generating_alpha, generating_gamma = round(generating_sweep_alphas[generating_param_idx].item(), 3), round(generating_sweep_gammas[generating_param_idx].item(), 3)
            generating_parameter_string = f"$\\alpha_g = {generating_alpha}, \gamma_g = {generating_gamma}$"

            for inferring_param_idx in range(num_infering_params):

                # Extract information that depends on inference (and therefore also possibly generation)
                inference_alpha, inference_gamma = round(inference_sweep_alphas[inferring_param_idx].item(), 3), round(inference_sweep_gammas[inferring_param_idx].item(), 3)
                inference_parameter_string = f"$\\alpha_i = {inference_alpha}, \gamma_i = {inference_gamma}$"
                repeated_synthetic_particle_mrw = all_synthetic_particle_residual_mrws[:,generating_param_idx,inferring_param_idx]
                inference_emissions_mrw = inference_distrubtion_synthetic_emissions_mrws[inferring_param_idx].item()
                
                actual_inference_emissions_distribution = inference_emissions_p_evals[inferring_param_idx]
                actual_inference_emissions_cdf = (actual_inference_emissions_distribution * (theta_axis[1] - theta_axis[0]).cpu().numpy()).cumsum()

                parameter_pair_string = f"{generating_parameter_string}, {inference_parameter_string}"

                for rep_idx in range(num_repetitions):
                    
                    # Extract all relevant info
                    synth_errors = all_synthetic_errors[rep_idx, generating_param_idx]
                    synth_error_order = np.argsort(synth_errors.flatten())
                    p_tilde_particle_weights = all_p_tilde_particle_weights[rep_idx, generating_param_idx, inferring_param_idx]


                    ordered_residual_particle_locations = synth_errors.flatten()[synth_error_order]                     # eps[mn] in order
                    ordered_residual_particle_weights = p_tilde_particle_weights.flatten()[synth_error_order]           # w[mn] in order of eps[mn]

                    b_emission_cdf_evals = approximate_cdf(actual_inference_emissions_cdf, theta_axis.cpu().numpy(), ordered_residual_particle_locations)    # b[mn] = F_p(eps[mn])
                    particle_cdf_evals_of_b_emission_cdf_evals = ordered_residual_particle_weights.cumsum()             # F_{p_b}(b[mn])
                    calibration_curve_alpha = 1.0 if (inferring_param_idx == generating_param_idx) else 0.35    # XXX change for crossinference

                    if pdb_flag:
                    #    import pdb; pdb.set_trace()
                        pdb_flag = False

                    axes[iN, 2].plot(b_emission_cdf_evals, particle_cdf_evals_of_b_emission_cdf_evals, alpha = calibration_curve_alpha)
                    ks_statistic = np.max(np.abs(b_emission_cdf_evals - particle_cdf_evals_of_b_emission_cdf_evals))
                    all_synth_ks_stats[rep_idx,generating_param_idx,inferring_param_idx] = ks_statistic


                    ### DIAGNOSTICS/VISUALISATION. XXX: come back to this and plot specific examples, with error bars across repetitions
                    if rep_idx > 0:
                        break
                    # Color shared between generation parameters - inferring parameters pairs
                    # 1a. Plot the actual CDF of the emission distribution
                    lines = axes[iN,0].plot(theta_axis, actual_inference_emissions_cdf, label = parameter_pair_string)
                    param_pair_color = lines[0].get_color()
                    # 1b. Plot the empirical CDF of the residual distribution
                    axes[iN,0].plot(ordered_residual_particle_locations, ordered_residual_particle_weights.cumsum(), alpha = 0.2, color = param_pair_color)

                    # 2. Same but for pdfs:
                    axes[iN,1].plot(theta_axis, actual_inference_emissions_distribution, label = f'generative emission rho = {round(synthetic_emissions_mrw, 3)}, inference emission rho = {round(inference_emissions_mrw, 3)} residual rho = {round(repeated_synthetic_particle_mrw.mean().item(), 3)}', alpha = 0.5, color = param_pair_color)
                    axes[iN,1].hist(synth_errors.flatten(), weights=p_tilde_particle_weights.flatten(), bins = 50, density = True, color = param_pair_color)


        if original_file_data_inference_flag:
            # 3. Same (ish) but for real:
            all_raw_dataset_errors = dg.all_errors.cpu().numpy()
            real_particle_mrw = round(bnp_mrvs[set_size], 3)
            real_p_tilde_particle_weights = recent_losses['residual_distribution']['particle_non_uniforn_weights'][set_size] / all_raw_dataset_errors.shape[0]
            axes[iN,-1].plot(theta_axis, inference_emissions_p_evals[0].flatten(), label = f'Emission rho = {real_particle_mrw}')
            axes[iN,-1].hist(all_raw_dataset_errors.flatten(), weights=real_p_tilde_particle_weights.flatten().cpu().numpy(), bins = 50, density = True, label = f'Residual rho = {round(real_particle_mrw, 3)}', histtype='step')

            # 4. Pure raw, no filtering out, just to show improvement
            actual_raw_dataset_errors = dg.all_errors[:,0]
            raw_mrvs = mean_resultant_length_from_angles(actual_raw_dataset_errors, torch.tensor(1.0))
            raw_ckrs = kurtosis_from_angles(actual_raw_dataset_errors, torch.tensor(1.0))
            axes[iN,-1].hist(actual_raw_dataset_errors.cpu().numpy(), bins = 50, density = True, label = f'raw, N={set_size}, particle rho = {round(raw_mrvs, 3)}', histtype='step')
            error_order = np.argsort(actual_raw_dataset_errors.cpu().numpy().flatten())
            cdf_same_weights_raw = np.ones_like(actual_raw_dataset_errors.flatten().cpu().numpy().cumsum())
            cdf_same_weights_raw = cdf_same_weights_raw / cdf_same_weights_raw.sum()


            #### Anyway back to KS statistics!
            all_raw_dataset_errors_order = np.argsort(all_raw_dataset_errors.flatten())
            real_ordered_residual_particle_locations = all_raw_dataset_errors.flatten()[all_raw_dataset_errors_order]
            real_ordered_residual_particle_weights = real_p_tilde_particle_weights.flatten().cpu().numpy()[all_raw_dataset_errors_order]

            all_real_ks_stats = np.zeros([num_infering_params])

            for real_parameter_index in range(num_infering_params):
                real_actual_inference_emissions_cdf = (inference_emissions_p_evals[real_parameter_index] * (theta_axis[1] - theta_axis[0]).cpu().numpy()).cumsum()
                real_b_emission_cdf_evals = approximate_cdf(real_actual_inference_emissions_cdf, theta_axis.cpu().numpy(), real_ordered_residual_particle_locations)    # b[mn] = F_p(eps[mn])
                real_particle_cdf_evals_of_b_emission_cdf_evals = real_ordered_residual_particle_weights.cumsum()             # F_{p_b}(b[mn])
                real_calibration_curve_alpha = 0.7 if real_parameter_index == 0 else 0.5        # first parameter is the data-fit one by convention
                label = "Data calibration with data-fit $\phi$" if real_parameter_index == 0 else "Data calibration with different $\phi$" if real_parameter_index == 1 else None
                axes[iN, 2].plot(real_b_emission_cdf_evals, real_particle_cdf_evals_of_b_emission_cdf_evals, color = 'black', alpha = real_calibration_curve_alpha, label = label)
                real_ks_statistic = np.max(np.abs(real_b_emission_cdf_evals - real_particle_cdf_evals_of_b_emission_cdf_evals))

                all_real_ks_stats[real_parameter_index] = real_ks_statistic

        # Stopped doing this
        #synth_betas = synthetic_data_circular_statistics_sweep['generated_data']['components'][set_size][0,idx]
        #correctly_selected_errors = [errors[beta - 1] for errors, beta in zip(synthetic_errors[0,idx], synth_betas)]
        #axes[2].hist(correctly_selected_errors, bins = 50, density = True, label = f'{save_name} correctly selected synthetic errors, N={set_size}', histtype='step')




    axes[0,0].set_title('CDF (particle opaque vs emissions translucent)')
    axes[0,1].set_title('PDF on synthetic data with same emission distribution')
    axes[0,2].set_title('Calibration for parameter pairs')
    axes[0,-1].set_title('PDF on real')

    for ax_row in axes:
        for ax in ax_row:
            ax.legend(fontsize = 6)

    plt.savefig(join(save_path, f"diagnostics_for_{code_name}.png"))

    all_synth_ks_stats_per_save_file[data_path] = all_synth_ks_stats    # [num_repetitions, num_generating_params, num_infering_params]
    all_real_ks_stats_per_save_file[data_path] = all_real_ks_stats      # [num_generating_params]
