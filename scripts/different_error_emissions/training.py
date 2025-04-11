import sys
from pathlib import Path

import warnings

from non_parametric_model.scripts.different_error_emissions.setup import *

from purias_utils.util.logging import configure_logging_paths
from purias_utils.util.plotting import legend_without_repeats, lighten_color

from purias_utils.error_modelling_torus.data_utils.loading_utils import dump_training_indices_to_path

from itertools import chain
import matplotlib.pyplot as plt


MINIBATCH_SIZE = args.M_batch_mini

lines_cmap = ['b','y','g','r','c','m','y','peru']
lines_cmap_alt = ['navy','orange','lime','maroon','teal','purple','gold','chocolate']


################ a little bit more setup ##############
Path(args.logging_base).mkdir(parents = True, exist_ok = True)
logging_directory = os.path.join(args.logging_base, args.run_name)
[training_print_path, testing_print_path], logging_directory, _ = configure_logging_paths(logging_directory, log_suffixes=["train", "all_subjects"], index_new=True)
dump_training_indices_to_path(dataset_generator, logging_directory)

parameter_save_path = os.path.join(logging_directory, '{model}.{ext}')

with open(training_print_path, 'a') as f:
    header_row = [
        "Progress",
        "batch_N",
        "batch_error_emissions",
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
        "error_emissions",
        "best_train_set_naive_likelihood"
    ]
    print(*header_row, sep = '\t', file=f)

args.write_to_yaml(os.path.join(logging_directory, 'args.yaml'))

with open(os.path.join(logging_directory, 'cmd.txt'), 'w') as f:
    print(*sys.argv, file = f)
#############################################



################ training loop - iteration method slightly different to original one ##############
t = -1

assert all(dg.M_test == 0 for dg in dataset_generator.data_generators.values()), "Havent figured out beta values yet!"

for batch_N, batch_M, deltas_batch, errors_batch, metadata_batch, batch_indices in dataset_generator.iterate_train_batches(
    dimensions = delta_dimensions, shuffle = True, total = T,
    metadata_selection_key = separation_metadata_name,
    metadata_selection_value = None
):

    batch_error_emissions_key = metadata_batch[separation_metadata_name][0][0].item()
    assert set(metadata_batch[separation_metadata_name][0].tolist()) == {batch_error_emissions_key}

    errors_batch = errors_batch.to('cuda')
    deltas_batch = deltas_batch.to('cuda')

    t+=1
    
    timer.loop_start()

    if args.flip_augmentation:
        augmentation_map_deltas = (2 * (torch.rand_like(deltas_batch) < 0.5).int()) - 1
        deltas_batch = deltas_batch * augmentation_map_deltas.to(dtype = deltas_batch.dtype, device = deltas_batch.device)

    training_info = swap_model.get_elbo_terms(error_emissions_key = batch_error_emissions_key, deltas = deltas_batch, data = errors_batch, max_variational_batch_size = MINIBATCH_SIZE)

    opt.zero_grad()

    selected_beta = batch_M / len(dataset_generator.data_generators[batch_N].limit_available_indices(separation_metadata_name, batch_error_emissions_key, False)[0])
    total_elbo = (training_info['total_log_likelihood'] - selected_beta * training_info['kl_term'])
    total_loss = - total_elbo

    distance_loss = torch.zeros(num_models).float().to(total_loss.device)
    total_loss = total_loss + args.distance_loss_weight * distance_loss

    total_loss.sum().backward()
    opt.step()

    torch.cuda.empty_cache()

    training_step_per_set_size[batch_N].append(t)
    training_step_per_error_emissions_key[batch_N][batch_error_emissions_key].append(t)

    batch_N_index = all_set_sizes.index(batch_N)   # set size index
    batch_eek_idx = error_emissions_keys_across_set_sizes.index(batch_error_emissions_key)
    all_kl_losses_per_set_size[t, :, batch_N_index] = training_info['kl_term'].detach().cpu()
    all_dist_losses_per_set_size[t, :, batch_N_index] = distance_loss.detach().cpu()[0]
    all_average_llh_losses_per_set_size[t, :, batch_N_index, batch_eek_idx] = training_info['total_log_likelihood'].detach().cpu()[0] / batch_M
    all_average_elbos_per_set_size[t, :, batch_N_index, batch_eek_idx] = total_elbo.detach().cpu()[0] / batch_M

    ### Scalar parameters plotted below...
    if D > 0:
        all_inverse_ells[t,:,batch_N_index] = swap_model.generative_model.swap_function.kernel_holder[str(batch_N)].inverse_ells.squeeze(1).squeeze(1).detach().cpu()
        all_scalers[t,:,batch_N_index] = swap_model.generative_model.swap_function.kernel_holder[str(batch_N)].scaler.squeeze(1).squeeze(1).detach().cpu()
    else:
        all_pi_s_tilde_means[t,:,batch_N_index] = swap_model.get_variational_model(batch_N).swap_logit_mean.squeeze(-1).detach().cpu()
        all_pi_s_tilde_stds[t,:,batch_N_index] = swap_model.get_variational_model(batch_N).swap_logit_std.squeeze(-1).detach().cpu()
    if include_pi_u_tilde:
        assert not remove_uniform
        all_pi_u_tildes[t,:,batch_N_index] = swap_model.generative_model.swap_function.pi_u_tilde_holder[str(batch_N)].pi_tilde.detach().cpu()
    if fix_non_swap and include_pi_1_tilde:
        all_pi_1_tildes[t,:,batch_N_index] = swap_model.generative_model.swap_function.pi_1_tilde_holder[str(batch_N)].pi_tilde.detach().cpu()
    all_emissions_parameters[t,:,batch_N_index,batch_eek_idx,:] = swap_model.generative_model.get_error_emissions(batch_error_emissions_key).emission_parameter(batch_N).detach().cpu()
    

    if (t % testing_frequency) == 0:
        test_save_steps.append(t)

        assert M_test_per_set_size == 0, "Don't reccommend cross validation with split error emissions model right now"

        for iN, set_size in enumerate(all_set_sizes):

            dg = dataset_generator.data_generators[set_size]

            all_likelihoods_per_datapoint = torch.zeros(num_models, dg.all_deltas.shape[0])

            recent_component_priors[set_size] = []
            
            for iS, testing_eek in enumerate(error_emissions_keys_across_set_sizes):
                
                all_eek_indices = dg.all_metadata_inverted[separation_metadata_name][testing_eek]
                all_dataset_errors = dg.all_errors[:,all_eek_indices,:].to('cuda')
                all_dataset_relevant_deltas = dg.all_deltas[all_eek_indices][...,delta_dimensions].unsqueeze(0).repeat(swap_model.num_models, 1, 1, 1).to('cuda')    # [Q, M_s, N, D]

                with torch.no_grad():
                    try:
                        all_test_time_inference_info = swap_model.get_elbo_terms(testing_eek, all_dataset_relevant_deltas, all_dataset_errors, max_variational_batch_size=MINIBATCH_SIZE, return_kl=False)
                    except:
                        mb_slices = [slice(i * M_batch, (i+1) * M_batch) for i in range(all_dataset_relevant_deltas.shape[1] // M_batch + int(all_dataset_relevant_deltas.shape[1] % M_batch > 0))]

                        all_test_time_inference_info = {'likelihood_per_datapoint': [], 'priors': []}
                        
                        for mb_slice in mb_slices:
                            test_time_inference_info = swap_model.get_elbo_terms(testing_eek, all_dataset_relevant_deltas[:,mb_slice], all_dataset_errors[:,mb_slice], max_variational_batch_size=MINIBATCH_SIZE, return_kl=False)
                            [all_test_time_inference_info[k].append(test_time_inference_info[k]) for k in all_test_time_inference_info.keys()]
                        
                        for k in all_test_time_inference_info.keys():
                            all_test_time_inference_info[k] = torch.concat(all_test_time_inference_info[k], 1)

                all_likelihoods_per_datapoint[:,all_eek_indices] = all_test_time_inference_info['likelihood_per_datapoint'].to(all_likelihoods_per_datapoint.dtype).detach().cpu()         # each [Q, M_s]
                recent_naive_log_likelihoods[set_size][testing_eek] = all_test_time_inference_info['likelihood_per_datapoint'].detach().cpu().numpy()
                recent_component_priors[set_size].append(all_test_time_inference_info['priors'].cpu().numpy())
            
            recent_component_priors[set_size] = np.concatenate(recent_component_priors[set_size], 1)

            all_average_train_set_naive_log_likelihoods[len(test_save_steps)-1,iN] = \
                dg.separate_to_test_and_train(all_likelihoods_per_datapoint.cpu(), average_over_data = True)[0].max()   # get best train llh

            with open(testing_print_path, 'a') as f:
                new_print_row = [
                    f"{t + 1}/{T}",                                                                                                 # Progress
                    set_size,                                                                                                       # set_size
                    round(all_average_train_set_naive_log_likelihoods[len(test_save_steps)-1,iN].mean(), 6),                        # best_train_set_naive_likelihood
                ]
                print(*new_print_row, sep = '\t', file=f)

        recent_losses = {
            "recent_naive_log_likelihoods": recent_naive_log_likelihoods,   # This is for saving so all model repeats
        }
        np.save(os.path.join(logging_directory, "recent_losses.npy"), recent_losses)
        print()


    if (t % logging_frequency == 0) and t > 0:

        plt.close('all')
        torch.save(swap_model.state_dict(), parameter_save_path.format(model = 'swap_model', ext = 'mdl'))

        ################ visualise loss #############
        training_curves_save_path = os.path.join(logging_directory, 'ELBO_optimisation_losses.png')
        fig_losses, axes = plt.subplots(2, 2, figsize = (20, 20))

        axes[0,0].set_title('kl term (down)')
        axes[0,1].set_title('thresholded inducing point distance loss (down)')
        axes[1,0].set_title('Training quantities (up) - per item')
        axes[1,1].set_title('Various loglikelihood estimates - per item')

        for iN, ss in enumerate(all_set_sizes):
            
            set_size_training_steps = training_step_per_set_size[ss]

            axes[0,0].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,set_size_training_steps], all_kl_losses_per_set_size[set_size_training_steps,:,iN].T))), label = f'N={ss}', c = lines_cmap[iN])
            axes[0,1].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,set_size_training_steps], all_dist_losses_per_set_size[set_size_training_steps,:,iN].T))), label = f'N={ss}', c = lines_cmap[iN])

            for iS, eek in enumerate(error_emissions_keys_across_set_sizes):
                
                error_emissions_training_steps = training_step_per_error_emissions_key[ss][eek]
                axes[1,0].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,error_emissions_training_steps], all_average_llh_losses_per_set_size[error_emissions_training_steps,:,iN,iS].T))), label = None if iS else f'(Naive) llh, N={ss}', c = lines_cmap_alt[iN])
                axes[1,0].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,error_emissions_training_steps], all_average_elbos_per_set_size[error_emissions_training_steps,:,iN,iS].T))), label = None if iS else f'(Naive) llh, N={ss}', c = lines_cmap_alt[iN])

                axes[1,1].plot(test_save_steps, all_average_train_set_naive_log_likelihoods[:len(test_save_steps),:,iN], label = f'naive_log_likelihoods, train, N = {ss}', linestyle = '-')

        legend_without_repeats(axes[0,0])
        legend_without_repeats(axes[0,1])
        legend_without_repeats(axes[1,0])
        legend_without_repeats(axes[1,1])

        fig_losses.suptitle('For one repeat only!')
        fig_losses.savefig(training_curves_save_path)
        #############################################


        #### Visualise predicted pdf on first example in training set of each set size
        fig_pdf, axes_pdf = plt.subplots(1, num_set_sizes, figsize = (num_set_sizes * 5, 5), squeeze = False)
        axes_pdf = axes_pdf[0]
        
        for i_ss, set_size in enumerate(all_set_sizes):
            
            dg = dataset_generator.data_generators[set_size]
            
            for i_ee, ee_key in enumerate(error_emissions_keys_across_set_sizes):
                all_ee_key_indices = dg.all_metadata_inverted[separation_metadata_name][ee_key]
                example_deltas_batch = dg.all_deltas[all_ee_key_indices[:1]][...,delta_dimensions].unsqueeze(0).repeat(swap_model.num_models, 1, 1, 1).to('cuda')    # [Q, 1, N, D]
                example_target_zetas_batch = dataset_generator.data_generators[set_size].all_target_zetas[all_ee_key_indices[:1]].cuda()
                theta_axis, pdfs = swap_model.visualise_pdf_for_example(ee_key, example_deltas_batch, example_target_zetas_batch, 360)
                axes_pdf[i_ss].plot(theta_axis.cpu().numpy(), pdfs[0].cpu().numpy(), label = ee_key)
            
            axes_pdf[i_ss].set_title(f'N = {set_size}')
            axes_pdf[i_ss].legend(title = 'error emissions key')   

            display_pi_u_tilde = all_pi_u_tildes[training_step_per_set_size[set_size][-1],:,i_ss]
            display_pi_1_tilde = all_pi_1_tildes[training_step_per_set_size[set_size][-1],:,i_ss]

            grid_count = 100 if D == 1 else 50

            fig_surfaces, fig_surfaces_num_rows, fig_surfaces_num_cols = swap_model.visualise_variational_approximation(
                set_size = set_size, grid_count = grid_count,
                pi_u_tildes = display_pi_u_tilde, pi_1_tildes = display_pi_1_tilde, 
                all_deltas = dataset_generator.data_generators[set_size].all_deltas[...,delta_dimensions].cpu().numpy(),
                recent_component_priors = recent_component_priors.get(set_size), true_mean_surface = true_mean_surfaces_dict[set_size], true_std_surface = true_std_surfaces_dict[set_size],
                min_separation= [min_separations[set_size][d] for d in delta_dimensions],
                max_separation= [max_separations[set_size][d] for d in delta_dimensions],
                deltas_label = [dataset_generator.feature_names[d] for d in delta_dimensions]
            )

            fig_surfaces.savefig(os.path.join(logging_directory, f'function_surface_{t}_{set_size}.png'))
        
        fig_pdf.suptitle('For one model repeat only!')
        fig_pdf.savefig(os.path.join(logging_directory, 'example_full_distribution.png'))
        #############################################


        ### plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,training_steps], all_kl_losses_per_set_size[training_steps,:,iN].T)))

        ################ visualise scalar parameters #############
        fig_scalar, axes = plt.subplots(2, 2, figsize = (10, 10))

        for iN, ss in enumerate(all_set_sizes):

            set_size_training_steps = training_step_per_set_size[ss]

            if D > 0:
                axes[1,0].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,set_size_training_steps], all_scalers[set_size_training_steps,:,iN].T))), label = f'N = {ss}', c = lines_cmap[iN], linestyle = '--')
            else:
                axes[1,0].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,set_size_training_steps], all_pi_s_tilde_means[set_size_training_steps,:,iN].T))), label = f'N = {ss}, mean', c = lines_cmap[iN], linestyle = '--')
                axes[1,0].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,set_size_training_steps], all_pi_s_tilde_stds[set_size_training_steps,:,iN].T))), label = f'N = {ss}, std', c = lines_cmap_alt[iN], linestyle = '--')

            for d in range(D):
                inverse_ell_color = lines_cmap[iN] if d == 0 else lines_cmap_alt[iN] if d == 1 else 'asdf'
                axes[0,0].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,set_size_training_steps], all_inverse_ells[set_size_training_steps,:,iN,d].T))), label = f'N = {ss}, d={d}', c = inverse_ell_color, linestyle = '--')
            if (not remove_uniform) and include_pi_u_tilde:
                axes[0,1].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,set_size_training_steps], all_pi_u_tildes[set_size_training_steps,:,iN].T))), label = '$\\tilde{\pi}_u$' + f', N = {ss}', c = lines_cmap_alt[iN])
            if fix_non_swap and include_pi_1_tilde:
                axes[0,1].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,set_size_training_steps], all_pi_1_tildes[set_size_training_steps,:,iN].T))), label = '$\\tilde{\pi}_1$' + f', N = {ss}', c = lines_cmap_alt[iN])

        axes[0,0].set_title('Kernel (inverse) length')
        axes[0,1].set_title('$\\tilde{\pi}_.$')
        axes[1,0].set_title('Kernel scaler')

        # fig_residual_deltas = plot_emission_distribution(args.emission_type, recent_delta_distributions, all_concentrations, generative_model, axes.flatten()[-1], device)    # XXX: redo for particle emissions
        # if fig_residual_deltas is not None:
        #     fig_residual_deltas.savefig(os.path.join(logging_directory, 'residual_deltas_hist.png'))# .format(t = t))

        emissions_param_axes = axes.flatten()[-1]

        for iN, ss in enumerate(all_set_sizes):
            for iS, eek in enumerate(error_emissions_keys_across_set_sizes):
                error_emissions_training_steps = training_step_per_error_emissions_key[ss][eek]
                param_color = None
                for iii in range(num_emissions_parameters):
                    param_color = emissions_param_axes.plot(
                        *list(chain.from_iterable(zip(scalar_plot_x_axis[:,error_emissions_training_steps], all_emissions_parameters[error_emissions_training_steps,:,iN,iS,iii].T))),
                        label = f'N = {ss}, param {iii}', c = lines_cmap[iN] if param_color is None else lighten_color(param_color, 0.6)
                    )[0].get_color()

        emissions_param_axes.set_title(
            ('Concentration' if emission_type == 'von_mises'
            else 'Alpha (first), Gamma (second)')
        )
        
        [legend_without_repeats(ax) for ax in axes.flatten()]
        fig_scalar.suptitle('For one repeat only!')
        fig_scalar.savefig(os.path.join(logging_directory, 'scalar_parameters.png'))# .format(t = t))
        ##########################################################



    #############################################
    elapsed_string, remaining_string = timer.loop_end()
    with open(training_print_path, 'a') as f:
        new_print_row = [
            f"{t + 1}/{T}",                                                                 # Progress
            batch_N,                                                                        # batch_N
            batch_error_emissions_key,                                                             # batch_error_emissions
            round(all_average_elbos_per_set_size[t,:,batch_N_index,batch_eek_idx].mean(), 6),             # avg_total_elbo
            round(all_average_llh_losses_per_set_size[t,:,batch_N_index,batch_eek_idx].mean(), 6),        # avg_llh_term
            round(all_kl_losses_per_set_size[t,:,batch_N_index].mean(), 6),        # kl_term
            round(all_dist_losses_per_set_size[t,:,batch_N_index].mean(), 6),               # distance_loss
        ]
        
        new_print_row.extend([elapsed_string, remaining_string])
        print(*new_print_row, sep = '\t', file=f)
    #############################################

