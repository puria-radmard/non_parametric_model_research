from os.path import join
import numpy as np
import matplotlib.pyplot as plt

from purias_utils.error_modelling_torus.data_utils.loading_utils import load_experimental_data, load_synthetic_data
from purias_utils.util.arguments_yaml import ConfigNamepace



def generate_binned_line_from_dg(dg, dim_delta, min_delta, errors_index = 0):

    if dim_delta == 'est':
        estim_dim_deltas = dg.all_deltas[...,1].cpu().numpy()    # [M, N, D] -> [M, N]
    elif dim_delta == 'cue':
        estim_dim_deltas = dg.all_deltas[...,0].cpu().numpy()    # [M, N, D] -> [M, N]
    errors = dg.all_errors[errors_index].cpu().numpy()     # [Q, M, N] -> [M, N]

    x = np.abs(estim_dim_deltas[:,1:].flatten())
    y = np.abs(errors[:,1:].flatten())

    nbins = 10
    bins = np.linspace(min_delta, np.pi, nbins + 1)
    n, _ = np.histogram(x, bins=bins)
    sy, _ = np.histogram(x, bins=bins, weights=y)
    sy2, bin_edges = np.histogram(x, bins=bins, weights=y*y)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)

    return x, y, bin_edges, mean, std


def get_relevant_min_delta_from_dg(dg, dim_delta):
    if dim_delta == 'est':
        estim_dim_deltas = dg.all_deltas[...,1].cpu().numpy()    # [M, N, D] -> [M, N]
    elif dim_delta == 'cue':
        estim_dim_deltas = dg.all_deltas[...,0].cpu().numpy()    # [M, N, D] -> [M, N]
    return np.min(np.abs(estim_dim_deltas[:,1:]))


def plot_over_subject_mean_lines_with_error_bars(bin_edgeses, means, xs, ys, axes, label = None, scatter = True, plot_individual_lines = True, **errorbar_kwargs):

    assert all([(be == bin_edgeses[0]).all() for be in bin_edgeses])

    all_lines = np.stack(means, 0)
    ranges = all_lines.std(0)
    total_mean = all_lines.mean(0)

    if scatter:
        for x, y in zip(xs, ys):
            axes.scatter(x, y, alpha = 0.05)

    # axes.scatter(x, y, label = label, alpha = 0.5)
    col = axes.errorbar((bin_edgeses[0][1:] + bin_edgeses[0][:-1])/2, total_mean, yerr=ranges, label = label, **errorbar_kwargs)[0].get_color()
    if plot_individual_lines:
        for line in all_lines:
            axes.plot((bin_edgeses[0][1:] + bin_edgeses[0][:-1])/2, line, alpha = 0.2, color = col)




if __name__ == '__main__':

    dataset_name = 'mcmaster2022_e2_dircue'
    stim_conditions = [
        'highC',
        'medC',
        'lowC',
    ]
    set_size = 4
    delta_min = np.pi / 3

    # dataset_name = 'mcmaster2022_e1_oricue'
    # stim_conditions = [
    #     'cue_AR3',
    #     'cue_AR2',
    #     'cue_AR1',
    # ]
    # set_size = 6
    # delta_min = np.pi / 6


    num_subjects = 10

    deltas_dim = 'cue'

    save_base_template = '/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_single_subject_summaries_7_10_24/{dataset_name}_{cond}/full_wrapped_stable_{sidx}'
    save_base_template_agg = '/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_summaries_7_10_24/{dataset_name}_{cond}/full_wrapped_stable_0'
    fig_path_base = '/homes/pr450/repos/research_projects/error_modelling_torus/results_link/precosyne_figures'

    figsize = 5
    fig, axes = plt.subplots(4, len(stim_conditions), figsize =(figsize * len(stim_conditions), 4 * figsize))

    for ax_col, scond in enumerate(stim_conditions):

        all_real_subject_bin_edges = []
        all_real_subject_mean = []
        all_real_xs = []
        all_real_ys = []
        all_synth_subject_bin_edges = []
        all_synth_subject_mean = []
        all_synth_xs = []
        all_synth_ys = []

        axes[0, ax_col].set_title(f'Real data, {scond}')
        axes[1, ax_col].set_title(f'Synth data, {scond}')
        axes[2, ax_col].set_title(f'Comparison, {scond}')

        for sidx in range(num_subjects):

            subject_save_base = save_base_template.format(dataset_name = dataset_name, cond = scond, sidx = sidx)
            args = ConfigNamepace.from_yaml_path(join(subject_save_base, 'args.yaml'))
            real_dataset_generator = load_experimental_data(args.dataset_name, None, None, 0, 0, 1, ConfigNamepace({'subjects': [sidx], 'stim_strengths': [scond]}), device = 'cpu') # one subject
            real_x, real_y, real_subject_bin_edges, real_subject_mean, _ = generate_binned_line_from_dg(real_dataset_generator.data_generators[set_size], deltas_dim, delta_min, errors_index = 0)
            all_real_xs.append(real_x)
            all_real_ys.append(real_y)

            placeholder_dataset_generator = load_experimental_data(args.dataset_name, None, None, 0, 0, args.num_models, args, device = 'cpu') # one subject again
            synthed_dataset_generator, *_ = load_synthetic_data(placeholder_dataset_generator, subject_save_base, 'single_subject_cross_fit')

            all_real_subject_bin_edges.append(real_subject_bin_edges)
            all_real_subject_mean.append(real_subject_mean)

            # for i in range(args.num_models):
            for i in [10]:
                synth_x, synth_y, synth_subject_bin_edges, synth_subject_mean, _ = generate_binned_line_from_dg(synthed_dataset_generator.data_generators[set_size], deltas_dim, delta_min, errors_index = i)

                all_synth_subject_bin_edges.append(synth_subject_bin_edges)
                all_synth_subject_mean.append(synth_subject_mean)
                all_synth_xs.append(synth_x)
                all_synth_ys.append(synth_y)

                break
                
        plot_over_subject_mean_lines_with_error_bars(all_real_subject_bin_edges, all_real_subject_mean, all_real_xs, all_real_ys, axes[0, ax_col])
        plot_over_subject_mean_lines_with_error_bars(all_synth_subject_bin_edges, all_synth_subject_mean, all_synth_xs, all_synth_ys, axes[1, ax_col])

        plot_over_subject_mean_lines_with_error_bars(all_real_subject_bin_edges, all_real_subject_mean, all_real_xs, all_real_ys, axes[2, ax_col], label = 'real', scatter = False)
        plot_over_subject_mean_lines_with_error_bars(all_synth_subject_bin_edges, all_synth_subject_mean, all_synth_xs, all_synth_ys, axes[2, ax_col], label = 'synth', scatter = False)

        axes[2, ax_col].legend()

        for ax in axes[:,ax_col]:
            ax.set_xlabel('response-distractor deviation')
            ax.set_ylabel(f'cued-distractor deviation in {deltas_dim} dimension')


        aggregated_save_base = save_base_template_agg.format(dataset_name = dataset_name, cond = scond)
        args = ConfigNamepace.from_yaml_path(join(aggregated_save_base, 'args.yaml'))
        real_dataset_generator = load_experimental_data(args.dataset_name, None, None, 0, 0, 1, ConfigNamepace({'subjects': None, 'stim_strengths': [scond]}), device = 'cpu') # one subject
        _, _, real_aggregated_bin_edges, real_aggregated_mean, _ = generate_binned_line_from_dg(real_dataset_generator.data_generators[set_size], deltas_dim, delta_min, errors_index = 0)
        placeholder_dataset_generator = load_experimental_data(args.dataset_name, None, None, 0, 0, args.num_models, args, device = 'cpu')
        synthed_dataset_generator, *_ = load_synthetic_data(placeholder_dataset_generator, aggregated_save_base, 'single_subject_cross_fit')
        synth_x, synth_y, synth_aggregated_bin_edges, synth_aggregated_mean, _ = generate_binned_line_from_dg(synthed_dataset_generator.data_generators[set_size], deltas_dim, delta_min, errors_index = 2)
        plot_over_subject_mean_lines_with_error_bars([real_aggregated_bin_edges], [real_aggregated_mean], None, None, axes[3, ax_col], label = 'real', scatter = False)
        plot_over_subject_mean_lines_with_error_bars([synth_aggregated_bin_edges], [synth_aggregated_mean], None, None, axes[3, ax_col], label = 'synth', scatter = False)
        axes[3, ax_col].set_title(f'Aggregated data, {scond}')
        



    fig_path = join(fig_path_base, f'straight_line_{deltas_dim}_dim_{dataset_name}.png')
    print(fig_path)
    fig.savefig(fig_path)

    plt.clf()
