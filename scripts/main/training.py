import sys
from pathlib import Path

from non_parametric_model.scripts.main.setup import *

from purias_utils.maths.circular_statistics import kurtosis_from_angles, mean_resultant_length_from_angles

from purias_utils.util.plotting import lighten_color, legend_without_repeats, standard_swap_model_simplex_plots

from purias_utils.util.logging import configure_logging_paths

from itertools import chain
import matplotlib.pyplot as plt

MINIBATCH_SIZE = args.M_batch_mini
try_residuals = True


lines_cmap = ['b','y','g','r','c','m','y','peru']
lines_cmap_alt = ['purple','orange']


################ a little bit more setup ##############
Path(args.logging_base).mkdir(parents = True, exist_ok = True)
logging_directory = os.path.join(args.logging_base, args.run_name)

print_path, logging_directory, _ = configure_logging_paths(logging_directory, index_new=True)
dump_training_indices_to_path(dataset_generator, logging_directory)

parameter_save_path = os.path.join(logging_directory, '{model}.{ext}')

import pdb; pdb.set_trace(header = 'Add importance sampled version here and elsewhere')
with open(print_path, 'a') as f:
    header_row = [
        "Progress",
        "batch_N",
        "avg_total_elbo",
        "avg_llh_term",
        "kl_term",
        "distance_loss",
        "avg_test_llh_term",
    ]
    print(*header_row, "elapsed", "remaining", sep = '\t', file=f)

args.set_size_to_M_train_each = dataset_generator.set_size_to_M_train_each

args.write_to_yaml(os.path.join(logging_directory, 'args.yaml'))

with open(os.path.join(logging_directory, 'cmd.txt'), 'w') as f:
    print(*sys.argv, file = f)
#############################################




################ training loop ##############
t = -1

for batch_N, batch_M, deltas_batch, errors_batch, batch_indices in dataset_generator.iterate_train_batches(dimensions = delta_dimensions, shuffle = True, total = T):

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

    training_step_per_set_size[batch_N].append(t)
    
    iN = all_set_sizes.index(batch_N)   # set size index
    all_kl_losses_per_set_size[t, :, iN] = training_info['kl_term'].detach().cpu()
    all_dist_losses_per_set_size[t, :, iN] = distance_loss.detach().cpu()
    all_average_llh_losses_per_set_size[t, :, iN] = training_info['total_log_likelihood'].detach().cpu() / batch_M
    all_average_elbos_per_set_size[t, :, iN] = total_elbo.detach().cpu() / batch_M

    # if M_batch < 1:
    #     #### Easy access to each repeat's performance on training set
    #     recent_train_sizes[batch_N] = batch_M
    #     recent_train_llh[batch_N] = training_info['total_log_likelihood'].detach().cpu().numpy()    # [Q]
    #     recent_component_posteriors[batch_N] = training_info['posterior'].detach().cpu().numpy()    # [Q, M, N+1]
    #     recent_component_priors[batch_N] = training_info['priors'].detach().cpu().numpy()           # [Q, M, N+1]

    if t % testing_frequency == 0:

        test_save_steps.append(t)

        import pdb; pdb.set_trace(header = 'Turn this into doing naive and importance sampled llh estimtes for all the items, then separating per training indices *just* for plotting/printing, but keeping it all together for recent_losses.json')

        #### Performance on test set

        with torch.no_grad():
            
            test_llh_terms, total_test_Ms = [{N: 0.0 for N in all_set_sizes} for _ in range(2)]
            total_test_llh = 0.0

            for test_batch_N, test_batch_M, test_deltas_batch, test_errors_batch, test_batch_indices in dataset_generator.all_test_batches(dimensions = delta_dimensions):

                test_info = swap_model.get_elbo_terms(test_deltas_batch, test_errors_batch, max_variational_batch_size=MINIBATCH_SIZE, return_kl=False)

                test_llh_terms[test_batch_N] += test_info['llh_term'].item()
                total_test_Ms[test_batch_N] += test_batch_M

        total_test_count = 0.0
        for iN, ss in enumerate(all_set_sizes):
            kN_test_count = total_test_Ms[ss] 
            new_llh = test_llh_terms[ss] / (kN_test_count if kN_test_count > 0 else 1)
            if kN_test_count == 0.0:
                new_llh = np.nan
            
            test_llh_losses_per_set_size[len(test_save_steps), :, iN] = new_llh

            total_test_count += kN_test_count

        if total_test_count == 0.0:
            total_test_llh = np.nan
        

    if t % testing_frequency == 0 and M_batch <= 0 and try_residuals:

        total_test_llh = np.nan
        total_test_count = np.nan

        #### Now iterate over all set sizes, and evaluate residual estimation for each of them
        for (set_size, dg) in dataset_generator.data_generators.items():

            all_dataset_errors = dg.all_errors                                                                                              # [Q, M, N]
            all_dataset_relevant_deltas = dg.all_deltas[...,delta_dimensions].unsqueeze(0).repeat(swap_model.num_models, 1, 1, 1)     # [Q, M, N, D]

            inference_info = swap_model.get_elbo_terms(all_dataset_relevant_deltas, all_dataset_errors, max_variational_batch_size=MINIBATCH_SIZE, return_kl=False)

            try:
                residual_estimation_weights_on_whole_dataset = swap_model.generative_model.empirical_residual_distribution_weights(inference_info['posterior'], all_dataset_errors)
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


    total_loss.sum().backward()
    opt.step()
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

        most_recent_emission_weights_and_locations = {}

        raise Exception('Update for general (uniform included) case')
        
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


    if (t % logging_frequency == 0):

        plt.close('all')
        torch.save(swap_model.state_dict(), parameter_save_path.format(model = 'swap_model', ext = 'mdl'))

        #if (args.emission_type == 'residual_deltas'):
        #    torch.save(most_recent_emission_weights_and_locations, parameter_save_path.format(model = 'generative_model_emission_histogram', ext = 'data'))



        ################ visualise loss #############
        training_curves_save_path = os.path.join(logging_directory, 'ELBO_optimisation_losses.png')
        fig_losses, axes = plt.subplots(2, 2, figsize = (20, 20))

        axes[0,0].set_title('kl term (down)')
        axes[0,1].set_title('thresholded inducing point distance loss (down)')
        axes[1,0].set_title('llh term (up) - averaged over displays')
        axes[1,1].set_title('total ELBO (up) - averaged over displays')

        for iN, ss in enumerate(all_set_sizes):

            training_steps = training_step_per_set_size[ss]

            axes[0,0].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,training_steps], all_kl_losses_per_set_size[training_steps,:,iN].T))), label = f'N={ss}', c = lines_cmap[iN])
            axes[0,1].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,training_steps], all_dist_losses_per_set_size[training_steps,:,iN].T))), label = f'N={ss}', c = lines_cmap[iN])
            axes[1,0].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,training_steps], all_average_llh_losses_per_set_size[training_steps,:,iN].T))), label = f'train N={ss}', c = lines_cmap[iN])
            axes[1,1].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,training_steps], all_average_elbos_per_set_size[training_steps,:,iN].T))), label = f'train N={ss}', c = lines_cmap[iN])
            
            if M_batch > 0:
                axes[1,0].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,training_steps], test_llh_losses_per_set_size[training_steps,:,iN].T))), label = f'test N={ss}', c = lines_cmap[iN])
        
        legend_without_repeats(axes[0,0])
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
                    axes[0,0].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,:t+1], all_inverse_ells[:t+1,:,iN,d].T))), label = f'N = {ss}, d={d}', c = lines_cmap[iN])
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

            grid_count = 100

            example_deltas_batch = dataset_generator.data_generators[set_size].all_deltas[[0]][...,delta_dimensions].unsqueeze(0).repeat(swap_model.generative_model.num_models, 1, 1, 1)
            example_target_zetas_batch = dataset_generator.data_generators[set_size].all_target_zetas[[0]]
            theta_axis, pdfs = swap_model.visualise_pdf_for_example(example_deltas_batch, example_target_zetas_batch, 360)
            for pdf in pdfs:
                axes_pdf[i_ss].plot(theta_axis.cpu().numpy(), pdf.cpu().numpy())
                axes_pdf[i_ss].set_title(f'N = {ss}')

            if swap_type == 'spike_and_slab':
                for q in range(swap_model.generative_model.num_models):
                    if set_size in recent_component_priors:
                        standard_swap_model_simplex_plots(recent_component_priors[set_size][q], axes_simplex[i_ss], ax_no_u = axes_simplex_no_u[i_ss])
                        legend_without_repeats(axes_simplex[i_ss])
                        legend_without_repeats(axes_simplex_no_u[i_ss])

            else:

                # if D == 2:
                #     fig_surfaces = plt.figure(figsize = (50, 30))

                #     axcuedslices_linear = fig_surfaces.add_subplot(2,4,3)
                #     axestimatedslices_linear = fig_surfaces.add_subplot(2,4,4)
                #     axcuedslices_exponentiated = fig_surfaces.add_subplot(2,4,7)
                #     axestimatedslices_exponentiated = fig_surfaces.add_subplot(2,4,8)
                    
                #     sliced_mean_surface_2D(
                #         surface, full_grid, swap_type, dataset_generator.feature_names[0], dataset_generator.feature_names[1],
                #         axcuedslices_linear, axestimatedslices_linear,
                #         axcuedslices_exponentiated, axestimatedslices_exponentiated,
                #         min_seperations[set_size], max_separations[set_size]
                #     )

                #     ax3d_linear = fig_surfaces.add_subplot(2,4,1, projection='3d')
                #     ax3d_exponentiated = fig_surfaces.add_subplot(2,4,5, projection='3d')
                #     axheat_linear = fig_surfaces.add_subplot(2,4,2, aspect=1.0)
                #     axheat_exponentiated = fig_surfaces.add_subplot(2,4,6, aspect=1.0)

                #     inducing_points = variational_model.Z.detach().cpu().T

                #     mean_and_variance_surface_2D(
                #         fig_surfaces, ax3d_linear, ax3d_exponentiated, axheat_linear, axheat_exponentiated, inducing_points,
                #         grid_x, grid_y, surface, upper_error_surface, lower_error_surface, 
                #         dataset_generator.feature_names[0], dataset_generator.feature_names[1]
                #     )                    

                # elif D == 1:

                display_pi_u_tilde = all_pi_u_tildes[t,:,i_ss]
                display_pi_1_tilde = all_pi_1_tildes[t,:,i_ss]

                fig_surfaces = swap_model.visualise_variational_approximation(
                    pi_u_tildes = display_pi_u_tilde, pi_1_tildes = display_pi_1_tilde, 
                    all_deltas = dataset_generator.data_generators[set_size].all_deltas[...,delta_dimensions].cpu().numpy(),
                    recent_component_priors = recent_component_priors.get(set_size), true_mean_surface = true_mean_surfaces_dict[set_size], true_std_surface = true_std_surfaces_dict[set_size],
                    min_separation= min_seperations[set_size][delta_dimensions[0]], max_separation=max_separations[set_size][delta_dimensions[0]],
                    deltas_label = dataset_generator.feature_names[delta_dimensions[0]]
                )

                fig_surfaces.savefig(os.path.join(logging_directory, f'function_surface_{t}_{set_size}.png'))

        fig_pdf.savefig(os.path.join(logging_directory, 'example_full_distribution.png'))
        #############################################

    elapsed_string, remaining_string = timer.loop_end()
    with open(print_path, 'a') as f:

        batch_N_index = all_set_sizes.index(batch_N)
        
        new_print_row = [
            f"{t + 1}/{T}",
            batch_N,
            round(all_average_elbos_per_set_size[t,:,batch_N_index].mean(), 6),
            round(all_average_llh_losses_per_set_size[t,:,batch_N_index].mean(), 6),
            round(all_kl_losses_per_set_size[t,:,batch_N_index].mean(), 6),
            round(all_dist_losses_per_set_size[t,:,batch_N_index].mean(), 6),
            round(total_test_llh / total_test_count, 6),
        ]

        new_print_row.extend([
            elapsed_string, remaining_string
        ])

        print(*new_print_row, sep = '\t', file=f)

    if M_batch < 1:

        recent_losses = {
            "residual_distribution": {
                "recent_naive_llh": recent_naive_llh,
                "recent_importance_sampled_llh": recent_importance_sampled_llh,
                "particle_non_uniforn_weights": recent_particle_non_uniform_weights,
                "particle_uniform_weights": recent_particle_uniform_weights,
                "particle_mean_first_resultant_vector_lengths": recent_particle_mean_first_resultant_vector_lengths,
                "particle_circular_kurtosis": recent_particle_circular_kurtosis
            },
        }

        np.save(os.path.join(logging_directory, "recent_losses.npy"), recent_losses)
