from non_parametric_model.commands.fc_single_subject_summaries.straight_line_est_dim import (
    generate_binned_line_from_dg, load_experimental_data, load_synthetic_data, plot_over_subject_mean_lines_with_error_bars,
    ConfigNamepace, np, plt, join
)
import matplotlib.lines as mlines
from non_parametric_model.commands.fc_single_subject_summaries import PALETTE

if __name__ == '__main__':

    fig, axes = plt.subplots(2, 2, figsize = (5.5, 7.2))

    information_cards = [
        dict(
            dataset_name = 'mcmaster2022_e2_dircue',
            stim_condition = 'medC',
            set_size = 4,
            delta_min = np.pi / 3,
        ),
        dict(
            dataset_name = 'mcmaster2022_e1_oricue',
            stim_condition = 'cue_AR1',
            set_size = 6,
            delta_min = np.pi / 6,
        ),
    ]

    synth_full_error_bar_offset = 0.03
    synth_sns_error_bar_offset = -0.03
    num_subjects = 10

    save_base_full_template = '/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_single_subject_summaries_7_10_24/{dataset_name}_{cond}/full_wrapped_stable_{sidx}'
    save_base_sns_template = '/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_single_subject_summaries_7_10_24/{dataset_name}_{cond}/spike_and_slab_wrapped_stable_{sidx}'
    fig_path_base = '/homes/pr450/repos/research_projects/error_modelling_torus/results_link/precosyne_figures'

    for ax_col, deltas_dim in zip(axes.T, ['cue', 'est']):
        
        for (ax, information_card) in zip(ax_col, information_cards):

            all_real_subject_bin_edges = []
            all_real_subject_mean = []
            all_synth_full_subject_bin_edges = []
            all_synth_sns_subject_bin_edges = []
            all_synth_full_subject_mean = []
            all_synth_sns_subject_mean = []
            all_real_aggregate_bin_edges = []
            all_real_aggregate_mean = []

            for sidx in range(num_subjects):

                subject_save_base_full = save_base_full_template.format(dataset_name = information_card['dataset_name'], cond = information_card['stim_condition'], sidx = sidx)
                subject_save_base_sns = save_base_sns_template.format(dataset_name = information_card['dataset_name'], cond = information_card['stim_condition'], sidx = sidx)
                args = ConfigNamepace.from_yaml_path(join(subject_save_base_full, 'args.yaml'))

                real_dataset_generator = load_experimental_data(args.dataset_name, None, None, 0, 0, 1, ConfigNamepace({'subjects': [sidx], 'stim_strengths': [information_card['stim_condition']]}), device = 'cpu') # one subject
                _, _, real_subject_bin_edges, real_subject_mean, _ = generate_binned_line_from_dg(real_dataset_generator.data_generators[information_card['set_size']], deltas_dim, information_card['delta_min'], errors_index = 0)
                all_real_subject_bin_edges.append(real_subject_bin_edges)
                all_real_subject_mean.append(real_subject_mean)

                placeholder_dataset_generator = load_experimental_data(args.dataset_name, None, None, 0, 0, args.num_models, args, device = 'cpu') # one subject again

                subject_all_synth_full_subject_mean = []
                subject_all_synth_sns_subject_mean = []

                for ei in range(16):
                
                    synthed_dataset_generator, *_ = load_synthetic_data(placeholder_dataset_generator, subject_save_base_full, 'single_subject_cross_fit')
                    _, _, ei_synth_full_subject_bin_edges, ei_synth_full_subject_mean, _ = generate_binned_line_from_dg(synthed_dataset_generator.data_generators[information_card['set_size']], deltas_dim, information_card['delta_min'], errors_index = ei)
                    subject_all_synth_full_subject_mean.append(ei_synth_full_subject_mean)

                    synthed_dataset_generator, *_ = load_synthetic_data(placeholder_dataset_generator, subject_save_base_sns, 'single_subject_cross_fit')
                    _, _, ei_synth_sns_subject_bin_edges, ei_synth_sns_subject_mean, _ = generate_binned_line_from_dg(synthed_dataset_generator.data_generators[information_card['set_size']], deltas_dim, information_card['delta_min'], errors_index = ei)
                    subject_all_synth_sns_subject_mean.append(ei_synth_sns_subject_mean)

                average_subject_all_synth_full_subject_mean = np.mean(subject_all_synth_full_subject_mean, 0)
                average_subject_all_synth_sns_subject_mean = np.mean(subject_all_synth_sns_subject_mean, 0)

                all_synth_full_subject_bin_edges.append(ei_synth_full_subject_bin_edges + synth_full_error_bar_offset)
                all_synth_full_subject_mean.append(average_subject_all_synth_full_subject_mean)
                all_synth_sns_subject_bin_edges.append(ei_synth_sns_subject_bin_edges + synth_sns_error_bar_offset)
                all_synth_sns_subject_mean.append(average_subject_all_synth_sns_subject_mean)

            plot_over_subject_mean_lines_with_error_bars(all_synth_full_subject_bin_edges, all_synth_full_subject_mean, None, None, ax, label = f'{information_card["dataset_name"]}, synth full, subject', scatter = False, plot_individual_lines = False, color = PALETTE.full, linewidth = 5)
            plot_over_subject_mean_lines_with_error_bars(all_synth_sns_subject_bin_edges, all_synth_sns_subject_mean, None, None, ax, label = f'{information_card["dataset_name"]}, synth sns, subject', scatter = False, plot_individual_lines = False, color = PALETTE.sns, linewidth = 5)
            plot_over_subject_mean_lines_with_error_bars(all_real_subject_bin_edges, all_real_subject_mean, None, None, ax, label = f'{information_card["dataset_name"]}, real, subject', scatter = False, plot_individual_lines = False, color = 'black')
        
            # ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([np.pi / 3, np.pi / 6, np.pi])
            ax.set_xticklabels(['$\pi / 3$', '$\pi / 6$', '$\pi$'], fontsize = 15)


    axes[1, 0].set_xlabel('Probe distance', fontsize = 15)
    axes[1, 1].set_xlabel('Spatial (report) distance', fontsize = 15)
    axes[0, 0].set_ylabel('Mean abs. dev. (dir. cued)', fontsize = 15)
    axes[1, 0].set_ylabel('Mean abs. dev. (ori. cued)', fontsize = 15)

    for ax_row in axes:

        ax_row[0].set_yticks([1.0, 2.0])
        ax_row[0].set_yticklabels([1.0, '$\pi$'], fontsize = 15)
        ax_row[1].set_yticks([])
        ymin = min([ax.get_ylim()[0] for ax in ax_row])
        ymax = max([ax.get_ylim()[1] for ax in ax_row])
        [ax.set_ylim(ymin, ymax) for ax in ax_row]
        ax_row[1].spines['left'].set_visible(False)

    label_synth_full = mlines.Line2D([], [], color=PALETTE.full, marker=None, linestyle='-', linewidth = 7, label='Full')
    label_synth_sns = mlines.Line2D([], [], color=PALETTE.sns, marker=None, linestyle='-', linewidth = 7, label='Simple')
    label_real = mlines.Line2D([], [], color='black', marker=None, linestyle='-', linewidth = 7, label='Real')
    fig.legend(handles=[label_real, label_synth_full, label_synth_sns], fontsize = 15, loc = 'upper center')


    fig_path = join(fig_path_base, f'final_figure_straight_lines.png')

    print(fig_path)
    fig.savefig(fig_path)
    fig.savefig(
        '/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/final_figure_straight_lines.svg',
        # bbox_inches='tight', 
        pad_inches=0, transparent=True, format = 'svg'
    )

    plt.clf()
