import sys
from pathlib import Path

from non_parametric_model.scripts.hierarchical.setup import *

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
        "batch_submodel",
        "avg_total_elbo",
        "avg_llh_term",
        "submodel_kl_term",
        "primary_kl_term",
        "distance_loss",
    ]
    print(*header_row, "elapsed", "remaining", sep = '\t', file=f)

with open(testing_print_path, 'a') as f:
    header_row = [
        "Progress",
        "set_size",
        "submodel",
        "avg_train_set_naive_likelihood"
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

    batch_submodel_key = metadata_batch[separation_metadata_name][0][0].item()
    assert set(metadata_batch[separation_metadata_name][0].tolist()) == {batch_submodel_key}

    errors_batch = errors_batch.to('cuda')
    deltas_batch = deltas_batch.to('cuda')

    t+=1
    
    timer.loop_start()

    if args.flip_augmentation:
        augmentation_map_deltas = (2 * (torch.rand_like(deltas_batch) < 0.5).int()) - 1
        deltas_batch = deltas_batch * augmentation_map_deltas.to(dtype = deltas_batch.dtype, device = deltas_batch.device)

    training_info = swap_model.get_elbo_terms(submodel_key = batch_submodel_key, deltas = deltas_batch, data = errors_batch, max_variational_batch_size = MINIBATCH_SIZE)

    opt.zero_grad()

    if num_models > 1:
        assert dataset_generator.data_generators[batch_N].M_test == 0, "submodel_beta doesnt work with more than one (set of) models being trained and not full train set!"
    submodel_beta = batch_M / len(dataset_generator.data_generators[batch_N].limit_available_indices(separation_metadata_name, batch_submodel_key, False)[0])
    primary_beta = batch_M / dataset_generator.data_generators[batch_N].M_train

    total_elbo = (training_info['total_log_likelihood'] - submodel_beta * training_info['submodel_kl_term'] - primary_beta * training_info['primary_kl_term'])
    total_loss = - total_elbo

    print('NEED TO REGULARISE ELL TILDE')

    distance_loss = torch.zeros(num_models).float().to(total_loss.device)
    total_loss = total_loss + args.distance_loss_weight * distance_loss

    total_loss.sum().backward()
    opt.step()

    torch.cuda.empty_cache()

    training_step_per_set_size[batch_N].append(t)
    training_step_per_submodel_key[batch_N][batch_submodel_key].append(t)

    batch_N_index = all_set_sizes.index(batch_N)   # set size index
    batch_smk_idx = submodel_keys_across_set_sizes.index(batch_submodel_key)
    all_primary_kl_losses_per_set_size[t, batch_N_index] = training_info['primary_kl_term'].detach().cpu()[0]
    all_submodel_kl_losses_per_set_size[t, batch_N_index, batch_smk_idx] = training_info['submodel_kl_term'].detach().cpu()[0]
    all_dist_losses_per_set_size[t, batch_N_index, batch_smk_idx] = distance_loss.detach().cpu()[0]
    all_average_llh_losses_per_set_size[t, batch_N_index, batch_smk_idx] = training_info['total_log_likelihood'].detach().cpu()[0] / batch_M
    all_average_elbos_per_set_size[t, batch_N_index, batch_smk_idx] = total_elbo.detach().cpu()[0] / batch_M


    ### Scalar parameters plotted below... XXX for one repeat only!!!
    all_primary_inverse_ells[t,batch_N_index] = swap_model.generative_model.primary_generative_model.swap_function.kernel_holder[str(batch_N)].inverse_ells.squeeze(1).squeeze(1).detach().cpu()[0]
    all_primary_scalers[t,batch_N_index] = swap_model.generative_model.primary_generative_model.swap_function.kernel_holder[str(batch_N)].scaler.squeeze(1).squeeze(1).detach().cpu()[0]
    all_submodel_inverse_ells[t,batch_N_index,batch_smk_idx] = swap_model.get_generative_submodel(batch_submodel_key).swap_function.kernel_holder[str(batch_N)].inverse_ells.squeeze(1).squeeze(1).detach().cpu()[0]
    all_submodel_scalers[t,batch_N_index,batch_smk_idx] = swap_model.get_generative_submodel(batch_submodel_key).swap_function.kernel_holder[str(batch_N)].scaler.squeeze(1).squeeze(1).detach().cpu()[0]
    if include_pi_u_tilde:
        assert not remove_uniform
        all_pi_u_tildes[t,batch_N_index,batch_smk_idx] = swap_model.get_generative_submodel(batch_submodel_key).swap_function.pi_u_tilde_holder[str(batch_N)].pi_tilde.detach().cpu()[0]
    if fix_non_swap and include_pi_1_tilde:
        all_pi_1_tildes[t,batch_N_index,batch_smk_idx] = swap_model.get_generative_submodel(batch_submodel_key).swap_function.pi_1_tilde_holder[str(batch_N)].pi_tilde.detach().cpu()[0]
    all_emissions_parameters[t,batch_N_index,batch_smk_idx] = swap_model.get_generative_submodel(batch_submodel_key).error_emissions.emission_parameter(batch_N).detach().cpu()[0]


    if (t % testing_frequency) == 0:
        test_save_steps.append(t)

        assert M_test_per_set_size == 0, "Don't reccommend cross validation with hierarchical model right now"

        for iN, set_size in enumerate(all_set_sizes):

            dg = dataset_generator.data_generators[set_size]

            all_likelihoods_per_datapoint = torch.zeros(num_models, dg.all_deltas.shape[0])
            
            for iS, testing_smk in enumerate(submodel_keys_across_set_sizes):
                
                all_smk_indices = dg.all_metadata_inverted[separation_metadata_name][testing_smk]
                all_dataset_errors = dg.all_errors[:,all_smk_indices,:].to('cuda')
                all_dataset_relevant_deltas = dg.all_deltas[all_smk_indices][...,delta_dimensions].unsqueeze(0).repeat(swap_model.num_models, 1, 1, 1).to('cuda')    # [Q, M_s, N, D]

                with torch.no_grad():
                    test_time_inference_info = swap_model.get_elbo_terms(testing_smk, all_dataset_relevant_deltas, all_dataset_errors, max_variational_batch_size=MINIBATCH_SIZE, return_kl=False)

                all_likelihoods_per_datapoint[:,all_smk_indices] = test_time_inference_info['likelihood_per_datapoint'].to(all_likelihoods_per_datapoint.dtype).detach().cpu()         # each [Q, M_s]
                recent_naive_log_likelihoods[set_size][testing_smk] = test_time_inference_info['likelihood_per_datapoint'].detach().cpu().numpy()
                recent_component_priors[set_size][testing_smk] = test_time_inference_info['priors'].cpu().numpy()

            all_average_train_set_naive_log_likelihoods[len(test_save_steps)-1,iN] = \
                dg.separate_to_test_and_train(all_likelihoods_per_datapoint.cpu(), average_over_data = True)[0][0] # This is just for plotting so just one model repeat

            with open(testing_print_path, 'a') as f:
                new_print_row = [
                    f"{t + 1}/{T}",                                                                                                 # Progress
                    set_size,                                                                                                       # set_size
                    round(all_average_train_set_naive_log_likelihoods[len(test_save_steps)-1,iN].mean(), 6),                        # avg_train_set_naive_likelihood
                ]
                print(*new_print_row, sep = '\t', file=f)

        recent_losses = {
            "recent_naive_log_likelihoods": recent_naive_log_likelihoods,   # This is for saving so all model repeats
        }
        np.save(os.path.join(logging_directory, "recent_losses.npy"), recent_losses)


    if (t % logging_frequency == 0):# and t > 0:

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
            
            primary_training_steps = training_step_per_set_size[ss]
            axes[0,0].plot(scalar_plot_x_axis[primary_training_steps], all_primary_kl_losses_per_set_size[primary_training_steps,iN], label = f'N={ss}, primary', c = lines_cmap[iN], linestyle = '--')

            for iS, smk in enumerate(submodel_keys_across_set_sizes):
                
                sub_training_steps = training_step_per_submodel_key[ss][smk]
                axes[0,0].plot(scalar_plot_x_axis[sub_training_steps], all_submodel_kl_losses_per_set_size[sub_training_steps,iN,iS], label = None if iS else f'N={ss}, sub', c = lines_cmap[iN])
                axes[0,1].plot(scalar_plot_x_axis[sub_training_steps], all_dist_losses_per_set_size[sub_training_steps,iN,iS], label = None if iS else f'N={ss}', c = lines_cmap[iN])
                axes[1,0].plot(scalar_plot_x_axis[sub_training_steps], all_average_llh_losses_per_set_size[sub_training_steps,iN,iS], label = None if iS else f'(Naive) llh, N={ss}', c = lines_cmap_alt[iN])
                axes[1,0].plot(scalar_plot_x_axis[sub_training_steps], all_average_elbos_per_set_size[sub_training_steps,iN,iS], label = None if iS else f'Total ELBO, N={ss}', c = lines_cmap[iN])
                axes[1,1].plot(test_save_steps, all_average_train_set_naive_log_likelihoods[:len(test_save_steps),iN], label = None if iS else f'naive_log_likelihoods, train, N = {ss}')

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
            
            for i_sub, sub_key in enumerate(submodel_keys_across_set_sizes):
                all_sub_key_indices = dg.all_metadata_inverted[separation_metadata_name][sub_key]
                example_deltas_batch = dg.all_deltas[all_sub_key_indices[:1]][...,delta_dimensions].unsqueeze(0).repeat(swap_model.num_models, 1, 1, 1).to('cuda')    # [Q, 1, N, D]
                example_target_zetas_batch = dataset_generator.data_generators[set_size].all_target_zetas[all_sub_key_indices[:1]].cuda()
                theta_axis, pdfs = swap_model.visualise_pdf_for_example(sub_key, example_deltas_batch, example_target_zetas_batch, 360)
                axes_pdf[i_ss].plot(theta_axis.cpu().numpy(), pdfs[0].cpu().numpy(), label = sub_key)
            
            axes_pdf[i_ss].set_title(f'N = {set_size}')
            axes_pdf[i_ss].legend(title = 'submodel key')   

            display_pi_u_tilde = all_pi_u_tildes[t,iN,:]
            display_pi_1_tilde = all_pi_1_tildes[t,iN,:]

            grid_count = 100 if D == 1 else 50

            fig_surfaces, fig_surfaces_num_rows, fig_surfaces_num_cols = swap_model.visualise_variational_approximation(
                set_size = set_size, grid_count = grid_count,
                pi_u_tildes = display_pi_u_tilde, pi_1_tildes = display_pi_1_tilde, 
                recent_component_priors = recent_component_priors, 
                true_primary_mean_surface=true_primary_surfaces[set_size], true_primary_std_surface=true_primary_surfaces_std[set_size], 
                true_submodel_mean_surface=true_submodel_surfaces[set_size], true_submodel_std_surface=true_submodel_surfaces_std[set_size], 
                true_total_mean_surface=true_total_surfaces[set_size], true_total_std_surface=true_total_surfaces_std[set_size],
                min_separation= [min_separations[set_size][d] for d in delta_dimensions],
                max_separation= [max_separations[set_size][d] for d in delta_dimensions],
                deltas_label = [dataset_generator.feature_names[d] for d in delta_dimensions],
                all_deltas = dataset_generator.data_generators[set_size].all_deltas[...,delta_dimensions].cpu().numpy(),
                indices_per_submodel = dg.all_metadata_inverted[separation_metadata_name],
            )     

            fig_surfaces.suptitle('For one repeat only!')
            fig_surfaces.savefig(os.path.join(logging_directory, f'function_surface_{t}_{set_size}.png'))
        
        fig_pdf.suptitle('For one repeat only!')
        fig_pdf.savefig(os.path.join(logging_directory, 'example_full_distribution.png'))
        #############################################



        ################ visualise scalar parameters #############
        fig_scalar, axes = plt.subplots(2, 2, figsize = (10, 10))

        for iN, ss in enumerate(all_set_sizes):

            primary_training_steps = training_step_per_set_size[ss]

            axes[1,0].plot(scalar_plot_x_axis[primary_training_steps], all_primary_scalers[primary_training_steps,iN], label = f'N = {ss}, primary', c = lines_cmap[iN], linestyle = '--')

            for d in range(D):
                inverse_ell_color = lines_cmap[iN] if d == 0 else lines_cmap_alt[iN] if d == 1 else 'asdf'
                axes[0,0].plot(scalar_plot_x_axis[primary_training_steps], all_primary_inverse_ells[primary_training_steps,iN,d], label = f'N = {ss}, d={d}, primary', c = inverse_ell_color, linestyle = '--')

            for iS, smk in enumerate(submodel_keys_across_set_sizes):

                sub_training_steps = training_step_per_submodel_key[ss][smk]

                for d in range(D):
                    inverse_ell_color = lines_cmap[iN] if d == 0 else lines_cmap_alt[iN] if d == 1 else 'asdf'
                    axes[0,0].plot(scalar_plot_x_axis[sub_training_steps], all_submodel_inverse_ells[sub_training_steps,iN,iS,d], label = None if iS else f'N = {ss}, d={d}, sub', c = inverse_ell_color)

                axes[1,0].plot(scalar_plot_x_axis[sub_training_steps], all_submodel_scalers[sub_training_steps,iN,iS], label = None if iS else f'N = {ss}, sub', c = lines_cmap[iN])

                if (not remove_uniform) and include_pi_u_tilde:
                    axes[0,1].plot(scalar_plot_x_axis[sub_training_steps], all_pi_u_tildes[sub_training_steps,iN,iS], label = '$\\tilde{\pi}_u$' + f', N = {ss}', c = lines_cmap_alt[iN])
                if fix_non_swap and include_pi_1_tilde:
                    axes[0,1].plot(scalar_plot_x_axis[sub_training_steps], all_pi_1_tildes[sub_training_steps,iN,iS], label = '$\\tilde{\pi}_1$' + f', N = {ss}', c = lines_cmap_alt[iN])

        axes[0,0].set_title('Kernel (inverse) length')
        axes[0,1].set_title('$\\tilde{\pi}_.$')
        axes[1,0].set_title('Kernel scaler')

        # fig_residual_deltas = plot_emission_distribution(args.emission_type, recent_delta_distributions, all_concentrations, generative_model, axes.flatten()[-1], device)    # XXX: redo for particle emissions
        # if fig_residual_deltas is not None:
        #     fig_residual_deltas.savefig(os.path.join(logging_directory, 'residual_deltas_hist.png'))# .format(t = t))

        emissions_param_axes = axes.flatten()[-1]

        for iN, ss in enumerate(all_set_sizes):
            for iS, smk in enumerate(submodel_keys_across_set_sizes):
                sub_training_steps = training_step_per_submodel_key[ss][smk]
                param_color = None
                for iii in range(num_emissions_parameters):
                    param_color = emissions_param_axes.plot(
                        scalar_plot_x_axis[sub_training_steps], all_emissions_parameters[sub_training_steps,iN,iS,iii], 
                        label = None if iS else f'N = {ss}, param {iii}', 
                        c = lines_cmap[iN] if param_color is None else lighten_color(param_color, 0.6)
                    )[0].get_color()

        emissions_param_axes.set_title(
            ('Concentration' if emission_type == 'von_mises'
            else 'Alpha (first), Gamma (second)') + " (all sub)"
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
            batch_submodel_key,                                                             # batch_submodel
            round(all_average_elbos_per_set_size[t,batch_N_index,batch_smk_idx].mean(), 6),             # avg_total_elbo
            round(all_average_llh_losses_per_set_size[t,batch_N_index,batch_smk_idx].mean(), 6),        # avg_llh_term
            round(all_submodel_kl_losses_per_set_size[t,batch_N_index,batch_smk_idx].mean(), 6),        # submodel_kl_term
            round(all_primary_kl_losses_per_set_size[t,batch_N_index].mean(), 6),         # primary_kl_term
            round(all_dist_losses_per_set_size[t,batch_N_index,batch_smk_idx].mean(), 6),               # distance_loss
        ]
        
        new_print_row.extend([elapsed_string, remaining_string])
        print(*new_print_row, sep = '\t', file=f)
    #############################################


