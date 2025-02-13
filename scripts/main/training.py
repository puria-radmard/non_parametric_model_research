import sys
from pathlib import Path

from non_parametric_model.scripts.main.setup import *

from purias_utils.maths.circular_statistics import kurtosis_from_angles, mean_resultant_length_from_angles

from purias_utils.util.plotting import lighten_color, legend_without_repeats, standard_swap_model_simplex_plots
from purias_utils.util.logging import configure_logging_paths

from purias_utils.error_modelling_torus.data_utils.loading_utils import dump_training_indices_to_path

from itertools import chain
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

MINIBATCH_SIZE = args.M_batch_mini
try_residuals = True


lines_cmap = ['b','y','g','r','c','m','y','peru']
lines_cmap_alt = ['navy','orange','lime','maroon','teal','purple','gold','chocolate']


################ a little bit more setup ##############
Path(args.logging_base).mkdir(parents = True, exist_ok = True)
logging_directory = os.path.join(args.logging_base, args.run_name)
[training_print_path, testing_print_path], logging_directory, _ = configure_logging_paths(logging_directory, log_suffixes=["train", "full"], index_new=True)
dump_training_indices_to_path(dataset_generator, logging_directory)

parameter_save_path = os.path.join(logging_directory, '{model}.{ext}')

with open(training_print_path, 'a') as f:
    header_row = [
        "Progress",
        "batch_N",
        "avg_total_elbo",
        "avg_llh_term",
        "kl_term",
        "distance_loss",
    ]
    print(*header_row, "elapsed", "remaining", sep = '\t', file=f)

with open(testing_print_path, 'a') as f:
    header_row = [
        "Progress",
        "set_size",
        "avg_train_set_naive_likelihood",
        "avg_test_set_naive_likelihood",
        "avg_train_set_importance_sampled_likelihood",
        "avg_test_set_importance_sampled_likelihood",
    ]
    if track_fmse:
        header_row.append("scaled_W_distance_on_distance")
    print(*header_row, sep = '\t', file=f)

args.dict['set_size_to_M_train_each'] = ConfigNamepace({str(k): v for k, v in dataset_generator.set_size_to_M_train_each.items()})

args.write_to_yaml(os.path.join(logging_directory, 'args.yaml'))

with open(os.path.join(logging_directory, 'cmd.txt'), 'w') as f:
    print(*sys.argv, file = f)
#############################################



################ training loop ##############
t = -1

for batch_N, batch_M, deltas_batch, errors_batch, *_ in dataset_generator.iterate_train_batches(dimensions = delta_dimensions, shuffle = True, total = T):

    errors_batch = errors_batch.to('cuda')
    deltas_batch = deltas_batch.to('cuda')

    t+=1
    
    timer.loop_start()

    if args.flip_augmentation:
        augmentation_map_deltas = (2 * (torch.rand_like(deltas_batch) < 0.5).int()) - 1
        deltas_batch = deltas_batch * augmentation_map_deltas.to(dtype = deltas_batch.dtype, device = deltas_batch.device)

        augmentation_map_errors = (2 * (torch.rand_like(errors_batch) < 0.5).int()) - 1
        errors_batch = errors_batch * augmentation_map_errors.to(dtype = errors_batch.dtype, device = deltas_batch.device)

    training_info = swap_model.get_elbo_terms(deltas = deltas_batch, data = errors_batch, max_variational_batch_size = MINIBATCH_SIZE)

    opt.zero_grad()

    selected_beta = betas[batch_N] if beta == 'nat' else beta
    total_elbo = (training_info['total_log_likelihood'] - selected_beta * training_info['kl_term'])
    total_loss = - total_elbo

    distance_loss = torch.relu(training_info['distance_loss'] - args.distance_loss_threshold)
    total_loss += args.distance_loss_weight * distance_loss

    total_loss.sum().backward()
    opt.step()

    torch.cuda.empty_cache()

    training_step_per_set_size[batch_N].append(t)
    
    batch_N_index = all_set_sizes.index(batch_N)   # set size index
    all_kl_losses_per_set_size[t, :, batch_N_index] = training_info['kl_term'].detach().cpu()
    all_dist_losses_per_set_size[t, :, batch_N_index] = distance_loss.detach().cpu()
    all_average_llh_losses_per_set_size[t, :, batch_N_index] = training_info['total_log_likelihood'].detach().cpu() / batch_M
    all_average_elbos_per_set_size[t, :, batch_N_index] = total_elbo.detach().cpu() / batch_M
    
    if t % testing_frequency == 0 and t > 0:

        test_save_steps.append(t)

        for iN, set_size in enumerate(all_set_sizes):
            
            dg = dataset_generator.data_generators[set_size]
            all_dataset_errors = dg.all_errors.to('cuda')                                                                                              # [Q, M, N]
            all_dataset_relevant_deltas = dg.all_deltas[...,delta_dimensions].unsqueeze(0).repeat(swap_model.num_models, 1, 1, 1).to('cuda')           # [Q, M, N, D]
            with torch.no_grad():
                test_time_inference_info = swap_model.get_elbo_terms(all_dataset_relevant_deltas, all_dataset_errors, max_variational_batch_size=MINIBATCH_SIZE, return_kl=False)
                # if swap_type != 'spike_and_slab':
                #     test_time_likelihood_estimates = swap_model.refined_likelihood_estimate(all_dataset_errors, all_dataset_relevant_deltas, dg.train_indices, max_variational_batch_size=MINIBATCH_SIZE) 
                # else:
                #     test_time_likelihood_estimates = {'importance_sampled_log_likelihoods': torch.nan * torch.ones_like(test_time_inference_info['likelihood_per_datapoint'])}

            recent_naive_log_likelihoods[set_size] = test_time_inference_info['likelihood_per_datapoint']          # each [Q, M]
            # recent_importance_sampled_log_likelihoods[set_size] = test_time_likelihood_estimates['importance_sampled_log_likelihoods']
            recent_component_priors[set_size] = test_time_inference_info['priors'].cpu().numpy()

            all_average_train_set_naive_log_likelihoods[len(test_save_steps)-1,:,iN], all_average_test_set_naive_log_likelihoods[len(test_save_steps)-1,:,iN] = \
                dg.separate_to_test_and_train(recent_naive_log_likelihoods[set_size].cpu(), average_over_data = True)
            # all_average_train_set_importance_sampled_log_likelihoods[len(test_save_steps)-1,:,iN], all_average_test_set_importance_sampled_log_likelihoods[len(test_save_steps)-1,:,iN] = \
            #     dg.separate_to_test_and_train(recent_importance_sampled_log_likelihoods[set_size].cpu(), average_over_data = True)

            if track_fmse:
                assert fix_non_swap
                print('track_fmse requires fix_non_swap for now!')

                with torch.no_grad():
                    scaled_fmse = (
                        np.square(test_time_inference_info['mean_surface'].cpu()[...,1:] - true_mean_surfaces_dict[set_size][...,1:]) +
                        (
                            test_time_inference_info['std_surface'].cpu().square()[...,1:] + np.square(true_std_surfaces_dict[set_size])[...,1:]
                            - 2*(test_time_inference_info['std_surface'].cpu()[...,1:] * true_std_surfaces_dict[set_size][...,1:])
                        )
                    )
                    aggregated_scaled_fmse = scaled_fmse.reshape(scaled_fmse.shape[0], -1).sqrt().mean(1) # all terms here of shape [Q, M, N-1] -> [Q]
                    all_scaled_rmse_from_true_confusion_function[len(test_save_steps)-1,:,iN] = aggregated_scaled_fmse

            with open(testing_print_path, 'a') as f:
                new_print_row = [
                    f"{t + 1}/{T}",                                                                                                 # Progress
                    set_size,                                                                                                       # set_size
                    round(all_average_train_set_naive_log_likelihoods[len(test_save_steps)-1,:,iN].mean(), 6),                      # avg_train_set_naive_likelihood
                    round(all_average_test_set_naive_log_likelihoods[len(test_save_steps)-1,:,iN].mean(), 6),                       # avg_test_set_naive_likelihood
                    round(all_average_train_set_importance_sampled_log_likelihoods[len(test_save_steps)-1,:,iN].mean(), 6),         # avg_train_set_importance_sampled_likelihood
                    round(all_average_test_set_importance_sampled_log_likelihoods[len(test_save_steps)-1,:,iN].mean(), 6),          # avg_test_set_importance_sampled_likelihood
                ]
                if track_fmse:
                    new_print_row.append(
                        round(all_scaled_rmse_from_true_confusion_function[len(test_save_steps)-1,:,iN].mean(), 6)
                    )
                print(*new_print_row, sep = '\t', file=f)

            if try_residuals:
                try:
                    residual_estimation_weights_on_whole_dataset = swap_model.generative_model.empirical_residual_distribution_weights(test_time_inference_info['posterior'], all_dataset_errors)
                    new_mrv = mean_resultant_length_from_angles(all_dataset_errors, residual_estimation_weights_on_whole_dataset['particle_weights_total'])
                    new_ckr = kurtosis_from_angles(all_dataset_errors, residual_estimation_weights_on_whole_dataset['particle_weights_total'])

                    recent_particle_uniform_weights[set_size] = residual_estimation_weights_on_whole_dataset['particle_weights_uniform']        # [total M, N]
                    recent_particle_non_uniform_weights[set_size] = residual_estimation_weights_on_whole_dataset['particle_weights_non_uniform']    # [total M, N]
                    recent_particle_mean_first_resultant_vector_lengths[set_size] = new_mrv # float
                    recent_particle_circular_kurtosis[set_size] = new_ckr   # float
                except Exception as e:
                    print(f'Too much to evaluate particle residual! {e}')
                    try_residuals = False
                    break
            else:
                recent_particle_uniform_weights[set_size] = None
                recent_particle_non_uniform_weights[set_size] = None
                recent_particle_mean_first_resultant_vector_lengths[set_size] = None
                recent_particle_circular_kurtosis[set_size] = None

        recent_losses = {
            "recent_naive_log_likelihoods": recent_naive_log_likelihoods,
            # "recent_importance_sampled_log_likelihoods": recent_importance_sampled_log_likelihoods,
            "residual_distribution": {
                "particle_non_uniforn_weights": recent_particle_non_uniform_weights,
                "particle_uniform_weights": recent_particle_uniform_weights,
                "particle_mean_first_resultant_vector_lengths": recent_particle_mean_first_resultant_vector_lengths,
                "particle_circular_kurtosis": recent_particle_circular_kurtosis
            } if try_residuals else 'residuals too expensive to compute!',
        }

        np.save(os.path.join(logging_directory, "recent_losses.npy"), recent_losses)

    #############################################


    ### Scalar parameters plotted below...
    for iN, ss in enumerate(all_set_sizes):
        if swap_type == 'spike_and_slab':
            sands_pi_vector = swap_model.generative_model.swap_function.generate_pi_vectors(ss, 1)['pis'].detach().cpu()
            all_p_unifs[t,:,iN] = sands_pi_vector[:,0,0]
            all_p_corrects[t,:,iN] = sands_pi_vector[:,0,1]
            if ss > 1:
                all_p_swaps[t,:,iN] = sands_pi_vector[:,0,2]

        else:
            all_inverse_ells[t,:,iN] = swap_model.generative_model.swap_function.kernel_holder[str(ss)].inverse_ells.squeeze(1).squeeze(1).detach().cpu()
            all_scalers[t,:,iN] = swap_model.generative_model.swap_function.kernel_holder[str(ss)].scaler.squeeze(1).squeeze(1).detach().cpu()
            if include_pi_u_tilde:
                assert not remove_uniform
                all_pi_u_tildes[t,:,iN] = swap_model.generative_model.swap_function.pi_u_tilde_holder[str(ss)].pi_tilde.detach().cpu()
            if fix_non_swap and include_pi_1_tilde:
                all_pi_1_tildes[t,:,iN] = swap_model.generative_model.swap_function.pi_1_tilde_holder[str(ss)].pi_tilde.detach().cpu()

    
    if (args.emission_type == 'residual_deltas') and (t % args.residual_deltas_update_frequencey == 0):
        # Log and update residual deltas based on (recycled) result of inference...

        raise Exception('Update for general (uniform included) case')

        most_recent_emission_weights_and_locations = {}
        
        for ss in recent_delta_distributions.keys():
        
            # Cache current one first...
            recent_delta_distributions[ss].append(generative_model.error_emissions.get_current_distribution(ss))

            inference_locations = errors_batch.flatten().detach()
            inference_weights = component_posterior_vectors[:,1:].flatten().detach() / component_posterior_vectors.shape[0]    # All datapoints have same p_env!
            # Then use locations and weights from the training step (which is why this is only implemented for M_batch = 0 at the moment)
            generative_model.error_emissions.load_new_distribution(ss, inference_locations, inference_weights)

            most_recent_emission_weights_and_locations[ss] = {
                "inference_locations": inference_locations.cpu(),
                "inference_weights": inference_weights.cpu(),
            }

    elif (args.emission_type != 'residual_deltas'):
        for iN, ss in enumerate(all_set_sizes):
            all_emissions_parameters[t,:,iN,:] = swap_model.generative_model.error_emissions.emission_parameter(ss).detach().cpu()


    if (t % logging_frequency == 0) and t > 0:

        plt.close('all')
        torch.save(swap_model.state_dict(), parameter_save_path.format(model = 'swap_model', ext = 'mdl'))

        #if (args.emission_type == 'residual_deltas'):
        #    torch.save(most_recent_emission_weights_and_locations, parameter_save_path.format(model = 'generative_model_emission_histogram', ext = 'data'))


        ################ visualise loss #############
        training_curves_save_path = os.path.join(logging_directory, 'ELBO_optimisation_losses.png')
        fig_losses, axes = plt.subplots(2, 2, figsize = (20, 20))

        axes[0,0].set_title('kl term (down)')
        axes[0,1].set_title('thresholded inducing point distance loss (down)')
        axes[1,0].set_title('Training quantities (up) - per item')
        axes[1,1].set_title('Various loglikelihood estimates - per item')

        for iN, ss in enumerate(all_set_sizes):

            training_steps = training_step_per_set_size[ss]

            axes[0,0].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,training_steps], all_kl_losses_per_set_size[training_steps,:,iN].T))), label = f'N={ss}', c = lines_cmap[iN])
            axes[0,1].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,training_steps], all_dist_losses_per_set_size[training_steps,:,iN].T))), label = f'N={ss}', c = lines_cmap[iN])
            axes[1,0].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,training_steps], all_average_llh_losses_per_set_size[training_steps,:,iN].T))), label = f'(Naive) llh, N={ss}', c = lines_cmap[iN])
            axes[1,0].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,training_steps], all_average_elbos_per_set_size[training_steps,:,iN].T))), label = f'Total ELBO, N={ss}', c = lines_cmap[iN])

            train_col = axes[1,1].plot(test_save_steps, all_average_train_set_naive_log_likelihoods[:len(test_save_steps),:,iN], label = f'naive_log_likelihoods, train, N = {ss}', linestyle = '-')[0].get_color()
            axes[1,1].plot(test_save_steps, all_average_train_set_importance_sampled_log_likelihoods[:len(test_save_steps),:,iN], label = f'importance_sampled_log_likelihoods, train, N = {ss}', linestyle = '--', c = train_col)
            test_col = axes[1,1].plot(test_save_steps, all_average_test_set_naive_log_likelihoods[:len(test_save_steps),:,iN], label = f'naive_log_likelihoods, test, N = {ss}', linestyle = '-')[0].get_color()
            axes[1,1].plot(test_save_steps, all_average_test_set_importance_sampled_log_likelihoods[:len(test_save_steps),:,iN], label = f'importance_sampled_log_likelihoods, test, N = {ss}', linestyle = '--', c = test_col)

        legend_without_repeats(axes[0,0])
        legend_without_repeats(axes[0,1])
        legend_without_repeats(axes[1,0])
        legend_without_repeats(axes[1,1])

        fig_losses.savefig(training_curves_save_path)
        #############################################


        ################ visualise scalar parameters #############
        if swap_type == 'spike_and_slab':
            fig_scalar, axes = plt.subplots(1, 2, figsize = (10, 5))

            for iN, ss in enumerate(all_set_sizes):
                correct_col = axes[0].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,:t+1], all_p_corrects[:t+1,:,iN].T))), label = f'p(correct | N={ss})', c = lines_cmap[iN])[0].get_color()
                axes[0].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,:t+1], all_p_unifs[:t+1,:,iN].T))), label = f'p(unifs | N={ss})', c = lighten_color(correct_col, 0.5))
                axes[0].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,:t+1], all_p_swaps[:t+1,:,iN].T))), label = f'p(swap | N={ss})', c = lighten_color(correct_col, 1.2))

        else:
            fig_scalar, axes = plt.subplots(2, 2, figsize = (10, 10))

            for iN, ss in enumerate(all_set_sizes):
                for d in range(D):
                    inverse_ell_color = lines_cmap[iN] if d == 0 else lines_cmap_alt[iN] if d == 1 else 'asdf'
                    axes[0,0].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,:t+1], all_inverse_ells[:t+1,:,iN,d].T))), label = f'N = {ss}, d={d}', c = inverse_ell_color)
                axes[1,0].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,:t+1], all_scalers[:t+1,:,iN].T))), label = f'N = {ss}', c = lines_cmap[iN])
                if (not remove_uniform) and include_pi_u_tilde:
                    axes[0,1].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,:t+1], all_pi_u_tildes[:t+1,:,iN].T))), label = '$\\tilde{\pi}_u$' + f', N = {ss}', c = lines_cmap_alt[iN])
                if fix_non_swap and include_pi_1_tilde:
                    axes[0,1].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,:t+1], all_pi_1_tildes[:t+1,:,iN].T))), label = '$\\tilde{\pi}_1$' + f', N = {ss}', c = lines_cmap[iN])

            axes[0,0].set_title('Kernel (inverse) length')
            axes[0,1].set_title('$\\tilde{\pi}_.$')
            axes[1,0].set_title('Kernel scaler')

        # fig_residual_deltas = plot_emission_distribution(args.emission_type, recent_delta_distributions, all_concentrations, generative_model, axes.flatten()[-1], device)    # XXX: redo for particle emissions
        # if fig_residual_deltas is not None:
        #     fig_residual_deltas.savefig(os.path.join(logging_directory, 'residual_deltas_hist.png'))# .format(t = t))

        emissions_param_axes = axes.flatten()[-1]

        for iN, ss in enumerate(all_set_sizes):
            param_color = None
            for iii in range(num_emissions_parameters):
                param_color = emissions_param_axes.plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,:t+1], all_emissions_parameters[:t+1,:,iN,iii].T))), label = f'N = {ss}, param {iii}', c = lines_cmap[iN] if param_color is None else lighten_color(param_color, 0.6))[0].get_color()

        emissions_param_axes.set_title(
            'Concentration' if emission_type == 'von_mises'
            else 'Alpha (first), Gamma (second)'
        )
        
        [legend_without_repeats(ax) for ax in axes.flatten()]
        fig_scalar.savefig(os.path.join(logging_directory, 'scalar_parameters.png'))# .format(t = t))
        ##########################################################

        #### Visualise predicted pdf on first example in training set of each set size
        # Done alongside simplex plot for spike and slab models:
        if swap_type == 'spike_and_slab':
            fig_pdf, axes_pdf = plt.subplots(num_set_sizes, 3, figsize = (15, num_set_sizes * 5), squeeze = False)
            axes_simplex = axes_pdf[:,0]
            axes_simplex_no_u = axes_pdf[:,1]
            axes_pdf = axes_pdf[:,2]
        else:
            fig_pdf, axes_pdf = plt.subplots(1, num_set_sizes, figsize = (num_set_sizes * 5, 5), squeeze = False)
            axes_pdf = axes_pdf[0]

        ##########################################################
        for i_ss, set_size in enumerate(all_set_sizes):

            grid_count = 100 if D == 1 else 50

            example_deltas_batch = dataset_generator.data_generators[set_size].all_deltas[[0]][...,delta_dimensions].unsqueeze(0).repeat(swap_model.generative_model.num_models, 1, 1, 1).cuda()
            example_target_zetas_batch = dataset_generator.data_generators[set_size].all_target_zetas[[0]].cuda()
            theta_axis, pdfs = swap_model.visualise_pdf_for_example(example_deltas_batch, example_target_zetas_batch, 360)
            for pdf in pdfs:
                axes_pdf[i_ss].plot(theta_axis.cpu().numpy(), pdf.cpu().numpy())
            axes_pdf[i_ss].set_title(f'N = {set_size}')

            if swap_type == 'spike_and_slab':
                for q in range(swap_model.generative_model.num_models):
                    if set_size in recent_component_priors:
                        standard_swap_model_simplex_plots(recent_component_priors[set_size][q], axes_simplex[i_ss], ax_no_u = axes_simplex_no_u[i_ss])
                        legend_without_repeats(axes_simplex[i_ss])
                        legend_without_repeats(axes_simplex_no_u[i_ss])

            else:

                try:
                    display_pi_u_tilde = all_pi_u_tildes[t,:,i_ss]
                    display_pi_1_tilde = all_pi_1_tildes[t,:,i_ss]

                    fig_surfaces, fig_surfaces_num_rows, fig_surfaces_num_cols = swap_model.visualise_variational_approximation(
                        set_size = set_size, grid_count = grid_count,
                        pi_u_tildes = display_pi_u_tilde, pi_1_tildes = display_pi_1_tilde, 
                        all_deltas = dataset_generator.data_generators[set_size].all_deltas[...,delta_dimensions].cpu().numpy(),
                        recent_component_priors = recent_component_priors.get(set_size), true_mean_surface = true_mean_surfaces_dict[set_size], true_std_surface = true_std_surfaces_dict[set_size],
                        min_separation= [min_separations[set_size][d] for d in delta_dimensions],
                        max_separation= [max_separations[set_size][d] for d in delta_dimensions],
                        deltas_label = [dataset_generator.feature_names[d] for d in delta_dimensions]
                    )

                    if track_fmse:
                        fig_surfaces_spec = gridspec.GridSpec(fig_surfaces_num_rows, fig_surfaces_num_cols, fig_surfaces)
                        fmse_axes = fig_surfaces.add_subplot(fig_surfaces_spec[-1,-2:])
                        fmse_axes.plot(test_save_steps, all_scaled_rmse_from_true_confusion_function[:len(test_save_steps),:,iN], label = 'Avg. Wass distance of $p(f)$ marginals', color = 'blue')

                    fig_surfaces.savefig(os.path.join(logging_directory, f'function_surface_{t}_{set_size}.png'))
                
                except torch._C._LinAlgError as e:
                    print("Error while displaying swap function!")
                    print(e)


        fig_pdf.savefig(os.path.join(logging_directory, 'example_full_distribution.png'))
        #############################################

    #############################################
    elapsed_string, remaining_string = timer.loop_end()
    with open(training_print_path, 'a') as f:
        new_print_row = [
            f"{t + 1}/{T}",                                                                 # Progress
            batch_N,                                                                        # batch_N
            round(all_average_elbos_per_set_size[t,:,batch_N_index].mean(), 6),             # avg_total_elbo
            round(all_average_llh_losses_per_set_size[t,:,batch_N_index].mean(), 6),        # avg_llh_term
            round(all_kl_losses_per_set_size[t,:,batch_N_index].mean(), 6),                 # kl_term
            round(all_dist_losses_per_set_size[t,:,batch_N_index].mean(), 6),               # distance_loss
        ]
        new_print_row.extend([elapsed_string, remaining_string])
        print(*new_print_row, sep = '\t', file=f)
    #############################################

        if early_stopper.advise(t):
            print('Stopping early', file=f)
            break

