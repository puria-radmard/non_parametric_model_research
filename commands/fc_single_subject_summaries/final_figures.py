# 14.10.24: figures plan:
    # 1. Scatter plot of betas with modulation in each direction shown
    # 2. Cmap of e^{<f>} of full fit for real data, and for 
    # 3. "Straight line" fit

    # TODO:
        # 1. mcmaster e3 --> final figure_
        # 2. After llh debugged: full boxplots for e3 LLH/BIC


import os, torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from purias_utils.util.arguments_yaml import ConfigNamepace
from purias_utils.util.plotting import prepare_axes_for_simplex, scatter_to_simplex

from purias_utils.error_modelling_torus.non_parametric_error_model.setup_utils import setup_model_whole
from purias_utils.error_modelling_torus.data_utils.loading_utils import load_experimental_data

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles

from non_parametric_model.commands.fc_single_subject_summaries import PALETTE

# dataset_name = 'mcmaster2022_e2_dircue_medC'
# MIN_SEP = np.pi / 3
# SET_SIZE = 4

dataset_name = 'mcmaster2022_e1_oricue_cue_AR2'
MIN_SEP = np.pi / 6
SET_SIZE = 6


delta_dimensions = ...


def get_exp_surface_from_path(path, chosen_model, do_grid=True, min_sep=MIN_SEP, set_size=SET_SIZE, make_exp = False):
    
    result_args = ConfigNamepace.from_yaml_path(os.path.join(path, 'args.yaml'))
    result_args.update(ConfigNamepace({'all_set_sizes': [set_size], 'trainable_kernel_delta': False, 'num_variational_samples': 4096, 'num_importance_sampling_samples': 4096}))
    result_args.dict['resume_path'] = path
    
    swap_model, D, delta_dimensions = setup_model_whole(**result_args.dict, all_min_seps = None, device = 'cuda')
    swap_model.reduce_to_single_model(chosen_model)
    swap_model.cuda()

    if do_grid:
        grid_inference_info = swap_model.inference_on_grid(set_size=set_size, grid_count=60)
        mean_surface = grid_inference_info['mean_surface'][0]
        std_surface = grid_inference_info['std_surface'][0]
        all_grid_points = grid_inference_info['all_grid_points'][0,:,1,:]
        include_mask = (all_grid_points >= min_sep).all(-1)
        interesting_grid_points = all_grid_points[include_mask]
        if make_exp:
            interesting_surface = np.exp(mean_surface + 0.5 * std_surface)[include_mask]
        else:
            interesting_surface = mean_surface[include_mask]
        return swap_model, interesting_grid_points, interesting_surface, result_args
    
    else:
        assert not make_exp
        return swap_model




if __name__ == '__main__':

    real_data_fit_path = f'/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_summaries_7_10_24/{dataset_name}/full_wrapped_stable_0'
    synth_data_fit_path = f'/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_crossfits_8_10_24/{dataset_name}/fit_to_spike_and_slab_von_mises_and_uniform/full_von_mises_and_uniform_0'
    sns_real_data_fit_path = f'/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_summaries_7_10_24/{dataset_name}/spike_and_slab_wrapped_stable_0'
    synth_cue_data_fit_path = f'/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_crossfits_8_10_24/{dataset_name}/fit_to_cue_dim_wrapped_stable/full_wrapped_stable_0'

    real_data_fit_swap_model, real_data_fit_interesting_grid_points, real_data_fit_interesting_exp_mean_surface, real_data_fit_result_args = get_exp_surface_from_path(real_data_fit_path, 1)
    sns_real_data_fit_swap_model = get_exp_surface_from_path(sns_real_data_fit_path, 0, False)
    if 'dircue' in dataset_name:
        synth_data_fit_swap_model, synth_data_fit_interesting_grid_points, synth_data_fit_interesting_exp_mean_surface, synth_data_fit_result_args = get_exp_surface_from_path(synth_data_fit_path, 2)
        synth_cue_data_fit_swap_model, synth_cue_data_fit_interesting_grid_points, synth_cue_data_fit_interesting_exp_mean_surface, synth_data_fit_result_args = get_exp_surface_from_path(synth_cue_data_fit_path, 9)
        selected_sorted_p_swap_diffs = [46, 212, 848]
    else:
        # selected_sorted_p_swap_diffs = [46, 55, 701]
        selected_sorted_p_swap_diffs = []


    sns_fit_prior = sns_real_data_fit_swap_model.generative_model.swap_function.generate_pi_vectors(SET_SIZE, 1)['pis'].flatten().cpu().detach().numpy()
    maximum_swap_sns_fit_prior = np.array([[sns_fit_prior[1], sns_fit_prior[2], 1 - sns_fit_prior[1] - sns_fit_prior[2]]])




    dataset_generator = load_experimental_data(real_data_fit_result_args.dataset_name, real_data_fit_result_args.train_indices_seed, None, real_data_fit_result_args.M_batch, real_data_fit_result_args.M_test_per_set_size, 1, real_data_fit_result_args, device = 'cpu')
    dg = dataset_generator.data_generators[SET_SIZE]
    all_dataset_errors = dg.all_errors.to('cuda')                                                                                              # [Q, M, N]
    all_dataset_relevant_deltas = dg.all_deltas[...,delta_dimensions].unsqueeze(0).repeat(real_data_fit_swap_model.num_models, 1, 1, 1).to('cuda')
    with torch.no_grad():
        test_time_inference_info = real_data_fit_swap_model.get_elbo_terms(all_dataset_relevant_deltas, all_dataset_errors, max_variational_batch_size=32, return_kl=False)
    data_priors = test_time_inference_info['priors'].cpu().numpy()[0]
    maximum_swap_pi_vectors = np.zeros([data_priors.shape[0], 3])
    maximum_swap_pi_vectors[:,0] = data_priors[:,1]  # Correct
    maximum_swap_pi_vectors[:,1] = data_priors[:,2:].max(-1) # maximum swap
    maximum_swap_pi_vectors[:,2] = 1.0 - maximum_swap_pi_vectors.sum(-1)

    # sorted_p_swap_diff_idxs = (data_priors[:,2:].max(-1) - data_priors[:,2:].min(-1)).argsort()
    # selected_sorted_p_swap_diffs = [sorted_p_swap_diff_idxs[i] for i in [3, 500, -8]]
    selected_pdf_evals, selected_actual_estimates, selected_correct_answers = [], [], []
    for pdf_idx in selected_sorted_p_swap_diffs:
        theta_axis, pdf_eval = real_data_fit_swap_model.visualise_pdf_for_example(all_dataset_relevant_deltas[:,[pdf_idx]], zeta_targets_batch=dg.all_target_zetas[[pdf_idx]].to('cuda'))
        theta_axis, pdf_eval = theta_axis.cpu().numpy(), pdf_eval[0].cpu().numpy()
        selected_pdf_evals.append(pdf_eval)
        selected_actual_estimates.append(
            rectify_angles(dg.all_target_zetas[pdf_idx,0,0] + dg.all_errors[0,pdf_idx,0]).item()
        )
        selected_correct_answers.append(dg.all_target_zetas[pdf_idx,0,0].item())
    print(selected_sorted_p_swap_diffs)






    swap_cued_colors = np.array([cued_delta[likely_swap_idx] for likely_swap_idx, cued_delta in zip(all_dataset_errors[0].abs().argmin(-1).tolist(), all_dataset_relevant_deltas[0,:,:,0].abs().tolist())])
    swap_esti_colors = np.array([cued_delta[likely_swap_idx] for likely_swap_idx, cued_delta in zip(all_dataset_errors[0].abs().argmin(-1).tolist(), all_dataset_relevant_deltas[0,:,:,1].abs().tolist())])



    fig = plt.figure(constrained_layout = False, figsize = (15, 10))
    fig_surfaces_spec = gridspec.GridSpec(6, 6, fig)

    if 'dircue' in dataset_name:
        v_min = min(real_data_fit_interesting_exp_mean_surface.min(), synth_data_fit_interesting_exp_mean_surface.min())
        v_max = max(real_data_fit_interesting_exp_mean_surface.max(), synth_data_fit_interesting_exp_mean_surface.max())
    else:
        v_min = real_data_fit_interesting_exp_mean_surface.min()
        v_max = real_data_fit_interesting_exp_mean_surface.max()

    axes_surface = fig.add_subplot(fig_surfaces_spec[:3,:3])
    edge_size = int(real_data_fit_interesting_exp_mean_surface.shape[0]**0.5)
    axes_surface.imshow(
        # *interesting_grid_points.reshape(edge_size, edge_size, 2).transpose(2, 0, 1),
        real_data_fit_interesting_exp_mean_surface.reshape(edge_size, edge_size).T,
        extent = [MIN_SEP, np.pi, MIN_SEP, np.pi],
        origin='lower', vmin = v_min, vmax = v_max
    )
    axes_surface.set_xticks([MIN_SEP, np.pi])
    axes_surface.set_yticks([MIN_SEP, np.pi])
    axes_surface.set_xticklabels(['$\pi/3$\n(min. feature seperation)', '$\pi$'])
    axes_surface.set_yticklabels(['$\pi/3$', '$\pi$'])
    axes_surface.set_xlabel('Random dot movement direction (probe)', labelpad=-25)
    axes_surface.set_ylabel('Random dot location (report)', labelpad=-20)

    if 'dircue' in dataset_name:
        axes_surface_synth = fig.add_subplot(fig_surfaces_spec[0,3])
        edge_size = int(synth_data_fit_interesting_exp_mean_surface.shape[0]**0.5)
        axes_surface_synth.imshow(
            # *interesting_grid_points.reshape(edge_size, edge_size, 2).transpose(2, 0, 1),
            synth_data_fit_interesting_exp_mean_surface.reshape(edge_size, edge_size).T,
            extent = [MIN_SEP, np.pi, MIN_SEP, np.pi],
            origin='lower', vmin = v_min, vmax = v_max
        )
        axes_surface_synth.set_xticks([])
        axes_surface_synth.set_yticks([])


        axes_surface_synth_cue = fig.add_subplot(fig_surfaces_spec[1,3])
        edge_size = int(synth_cue_data_fit_interesting_exp_mean_surface.shape[0]**0.5)
        axes_surface_synth_cue.imshow(
            # *interesting_grid_points.reshape(edge_size, edge_size, 2).transpose(2, 0, 1),
            synth_cue_data_fit_interesting_exp_mean_surface.reshape(edge_size, edge_size).T,
            extent = [MIN_SEP, np.pi, MIN_SEP, np.pi],
            # origin='lower', vmin = v_min, vmax = v_max
        )
        axes_surface_synth_cue.set_xticks([])
        axes_surface_synth_cue.set_yticks([])

    axes_coverage = fig.add_subplot(fig_surfaces_spec[2,3])
    axes_coverage.scatter(*all_dataset_relevant_deltas[0,:,1:,:].reshape(-1, 2).abs().cpu().numpy().T, alpha = 0.4, marker = 'x', s=3, color = 'grey')
    axes_coverage.set_ylim()
    # axes_coverage.set_xticklabels(['$\pi/3$', '$\pi$'])
    # axes_coverage.set_yticklabels(['$\pi/3$', '$\pi$'])
    axes_coverage.set_aspect('equal')
    axes_coverage.set_xticks([])
    axes_coverage.set_yticks([])

    # # axes_simplex_cued = fig.add_subplot(fig_surfaces_spec[3:,:3], aspect = 1)
    # # prepare_axes_for_simplex(axes_simplex_cued, ['Correct', 'Largest\nSwap', 'Everything Else'], label_kwargs = {})
    # # scatter_to_simplex(axes_simplex_cued, maximum_swap_pi_vectors, alpha = 0.2, c=swap_cued_colors, s=3, cmap = 'inferno')
    # # scatter_to_simplex(axes_simplex_cued, maximum_swap_sns_fit_prior, alpha = 0.5, color = 'brown', marker = 'x')
    # # axes_coverage.set_aspect('equal')

    # # axes_simplex_esti = fig.add_subplot(fig_surfaces_spec[3:,3:], aspect = 1)
    # # prepare_axes_for_simplex(axes_simplex_esti, ['Correct', 'Largest\nSwap', 'Everything Else'], label_kwargs = {})
    # # scatter_to_simplex(axes_simplex_esti, maximum_swap_pi_vectors, alpha = 0.2, c=swap_esti_colors, s=3, cmap = 'inferno')
    # # scatter_to_simplex(axes_simplex_esti, maximum_swap_sns_fit_prior, alpha = 0.5, color = 'brown', marker = 'x')
    # # axes_coverage.set_aspect('equal')

    axes_simplex = fig.add_subplot(fig_surfaces_spec[3:,3:], aspect = 1)
    prepare_axes_for_simplex(axes_simplex, ['Correct', 'Largest\nSwap', 'Everything Else'], label_kwargs = {})
    scatter_to_simplex(axes_simplex, maximum_swap_pi_vectors, alpha = 0.2, color = 'grey', s=3)
    scatter_to_simplex(axes_simplex, maximum_swap_sns_fit_prior, alpha = 0.5, color = 'brown', marker = 'x')
    axes_simplex.set_aspect('equal')


    axes_pdf = fig.add_subplot(fig_surfaces_spec[:3,4:], aspect = 3.0)
    for col, pdf_idx, pdf_eval, actual_estimate, correct_answer in zip(PALETTE.examples, selected_sorted_p_swap_diffs, selected_pdf_evals, selected_actual_estimates, selected_correct_answers):    
        axes_pdf.plot(theta_axis, pdf_eval, color = col)
        axes_pdf.plot([actual_estimate], [1.30], color = col, marker = 'v')
        axes_pdf.plot([correct_answer], [1.45], color = col, marker = 'o')

        #scatter_to_simplex(axes_simplex_cued, maximum_swap_pi_vectors[[pdf_idx]], alpha = 0.8, color = col)
        #scatter_to_simplex(axes_simplex_esti, maximum_swap_pi_vectors[[pdf_idx]], alpha = 0.8, color = col)
        scatter_to_simplex(axes_simplex, maximum_swap_pi_vectors[[pdf_idx]], alpha = 0.8, color = col)
        axes_surface.scatter(*all_dataset_relevant_deltas[0,[pdf_idx],1:,:].reshape(-1, 2).abs().cpu().numpy().T, marker = 'o', s=50, color = col)

    axes_pdf.set_xlabel('Location')
    axes_pdf.set_ylim(0)
    axes_pdf.set_xlim(-np.pi, +np.pi)
    axes_pdf.set_xlabel('$p(y | Z)$')
    axes_pdf.spines['top'].set_visible(False)
    axes_pdf.spines['left'].set_visible(False)
    axes_pdf.spines['right'].set_visible(False)
    axes_pdf.get_yaxis().set_ticks([])

    fig.savefig(
        f'/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/final_figure_{dataset_name}.svg',
        # bbox_inches='tight', 
        pad_inches=0, transparent=True, format = 'svg'
    )
