import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
import math

from non_parametric_model.commands.fc_single_subject_summaries import PALETTE


from purias_utils.util.arguments_yaml import ConfigNamepace



results_base_single_subject = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_single_subject_summaries_7_10_24"
results_base_aggregated_subject = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_summaries_7_10_24"
figure_base = "/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/llh_comparison_figures"

dataset_name = 'mcmaster2022_e1_oricue'
# subject_indices = [1, 2, 4, 5]
subject_indices = range(10)
set_sizes = [6]
stim_conditions = ['_cue_AR1', '_cue_AR2', '_cue_AR3']

# dataset_name = 'mcmaster2022_e2_dircue'
# # subject_indices = [0, 2, 5, 9]
# subject_indices = range(10)
# set_sizes = [4]
# stim_conditions = ['_lowC', '_medC', '_highC']

# # dataset_name = 'schneegans2017_e2_cueOrientation_reportColor'
# dataset_name = 'schneegans2017_e2_cueColor_reportOrientation'
# subject_indices = []
# set_sizes = [6]
# stim_conditions = ['']

# dataset_name = 'vandenberg2012_color'
# subject_indices = range(10)
# set_sizes = list(range(1, 9))
# stim_conditions = ['']


model_names = [
    'spike_and_slab_von_mises_and_uniform',
    'spike_and_slab_wrapped_stable',
    'cue_dim_von_mises_and_uniform',
    'cue_dim_wrapped_stable',
    'est_dim_von_mises_and_uniform',
    'est_dim_wrapped_stable',
    'full_von_mises_and_uniform',
    'full_von_mises',
    'full_wrapped_stable',
]

BASELINE_MODEL_NAME = model_names[0]
BASELINE_MODEL_NAME_ALT = model_names[1]

FIG_HEIGHT = 3
FIG_ASPECT = 1.2
SHOWFLIERS=False


def extract_results(log_base_addr):
    results_path = os.path.join(log_base_addr, "recent_losses.npy")
    config_path = os.path.join(log_base_addr, "args.yaml")
    model_path = os.path.join(log_base_addr, "swap_model.mdl")
    log_path = os.path.join(log_base_addr, "epoch_log_train.csv")

    recreated_args = ConfigNamepace.from_yaml_path(config_path)
    assert not (recreated_args.shared_emission_distribution or recreated_args.shared_swap_function), "BIC calc cannot work with sharing across set sizes"
    assert recreated_args.M_test_per_set_size == 0, "Calculation for no test set case!"

    model_params = torch.load(model_path)
    param_counts = [v.numel() / recreated_args.num_models for k, v in model_params.items() if k.split('.')[0] == 'generative_model']
    assert all([pc == int(pc) for pc in param_counts])
    num_params = sum(param_counts)

    results = np.load(results_path, allow_pickle=True).item()

    num_lines = sum(1 for _ in open(log_path))
    assert num_lines > 10, "Log too short, run clearly failed!"

    return results, num_params


def generate_df_cats(set_size, results_dict, num_params, subject_code, model_code, scond):

    try:
        average_training_llh = results_dict['recent_naive_log_likelihoods'][set_size].mean(-1).cpu().numpy()
    except KeyError:
        print(log_base)
        return None, None
    best_training_llh = results_dict['recent_naive_log_likelihoods'][set_size].mean(-1).cpu().numpy().max()
    num_data = results_dict['recent_naive_log_likelihoods'][set_size].shape[1]

    bic_adjustment = num_params * math.log(num_data) / num_data

    df_cat_reps = pd.DataFrame(
        {
            'Train LLH - spread over training repeats': average_training_llh,
            'Train BIC - spread over training repeats': average_training_llh - bic_adjustment,
        }
    )
    df_cat_reps['Model name'] = model_code
    df_cat_reps['Subject number'] = subject_code
    df_cat_reps['Set size'] = set_size
    df_cat_reps['Stim. condition'] = scond
    df_cat_reps['num_params'] = num_params
    df_cat_reps['num_data'] = num_data
    df_cat_reps['Train BIC_adjustment - spread over training repeats'] = bic_adjustment

    df_cat_best = pd.DataFrame(
        {
            'Train LLH - spread over subjects': [best_training_llh],
            'Train BIC - spread over subjects': [best_training_llh - bic_adjustment],
        }
    )
    df_cat_best['Model name'] = model_code
    df_cat_best['Set size'] = set_size
    df_cat_best['Subject number'] = subject_code
    df_cat_best['Stim. condition'] = scond
    df_cat_best['num_params'] = num_params
    df_cat_best['num_data'] = num_data
    df_cat_best['Train BIC_adjustment - spread over subjects'] = bic_adjustment

    return df_cat_reps, df_cat_best
    

if __name__ == '__main__':

    assert (len(set_sizes) == 1 or len(stim_conditions) == 1)
    # if len(stim_conditions) == 1:
    #     assert stim_conditions[0] == ''


    df_reps = pd.DataFrame()
    df_best = pd.DataFrame()

    for stim_condition in stim_conditions:

        for model_name in model_names:
            
            ######## AGGREGATED
            log_base = os.path.join(results_base_aggregated_subject, f"{dataset_name}{stim_condition}", f"{model_name}_0")
            try:
                results, num_params = extract_results(log_base)
                for set_size in set_sizes:
                    df_cat_reps, df_cat_best = generate_df_cats(set_size, results, num_params, 'Aggregated', model_name, stim_condition)
                    if df_cat_reps is not None:
                        df_reps = pd.concat([df_reps, df_cat_reps], ignore_index=True)
                        df_best = pd.concat([df_best, df_cat_best], ignore_index=True)
            except (FileNotFoundError, AssertionError):
                print('MISSING', log_base)


            ######## NOT AGGREGATED
            for subject_index in subject_indices:

                log_base = os.path.join(results_base_single_subject, f"{dataset_name}{stim_condition}", f"{model_name}_{subject_index}")
                try:
                    results, num_params = extract_results(log_base)
                except (FileNotFoundError, AssertionError) as e:
                    print(f'MISSING ({e})', log_base)
                    continue

                for set_size in set_sizes:
                    df_cat_reps, df_cat_best = generate_df_cats(set_size, results, num_params, subject_index, model_name, stim_condition)
                    df_reps = pd.concat([df_reps, df_cat_reps], ignore_index=True)
                    df_best = pd.concat([df_best, df_cat_best], ignore_index=True)


    if False:

        print()

        col_sep = 'Set size' if len(stim_conditions) == 1 else 'Stim. condition'
        # sns.catplot(data=df_reps, kind='box', x='Subject number', y='val', hue='Model name', col=col_sep, height=FIG_HEIGHT, aspect=FIG_ASPECT, palette=None, sharey='row', showfliers=SHOWFLIERS)
        for stat_name in ['LLH', 'BIC', 'BIC_adjustment']:
            sns.catplot(data=df_reps, kind='box', col='Subject number', x=f'Train {stat_name} - spread over training repeats', y='Model name', row=col_sep, height=5, aspect=1.2, palette=None, sharey='row', sharex='none', showfliers=SHOWFLIERS, margin_titles=True)
            # plt.tight_layout()
            # plt.suptitle(f'{dataset_name}')
            savepath = os.path.join(figure_base, f'{stat_name.lower()}_over_training_repeats_{dataset_name}.png')
            plt.savefig(savepath, format='png')

            print(savepath)
        
        print()

        if len(subject_indices):

            baseline_best_df = df_best[df_best['Model name'] == BASELINE_MODEL_NAME]
            df_best_minus_baseline = df_best.merge(baseline_best_df,on=['Set size', 'Subject number', 'Stim. condition'],how='left')
            df_best_minus_baseline['Model name'] = df_best_minus_baseline['Model name_x']

            for stat_name in ['LLH', 'BIC', 'BIC_adjustment']:
                df_best_minus_baseline[f'Train {stat_name} - spread over subjects'] = df_best_minus_baseline[f'Train {stat_name} - spread over subjects_x'] - df_best_minus_baseline[f'Train {stat_name} - spread over subjects_y']

                sns.catplot(data=df_best_minus_baseline[df_best_minus_baseline['Subject number'] != 'Aggregated'], kind='box', x=f'Train {stat_name} - spread over subjects', y='Model name', row=col_sep, height=5, aspect=1.2, palette=None, sharey='row', sharex='none', showfliers=SHOWFLIERS, )
                plt.tight_layout()
                # plt.suptitle(f'{dataset_name}')
                savepath = os.path.join(figure_base, f'{stat_name.lower()}_over_subjects_{dataset_name}.png')
                plt.savefig(savepath, format='png')

                print(savepath)




    ############
    # For abstract
    ############
    #if dataset_name == 'mcmaster2022_e2_dircue':
    if 'mcmaster' in dataset_name:

        baseline_best_df = df_best[df_best['Model name'] == BASELINE_MODEL_NAME_ALT]
        df_best_minus_baseline = df_best.merge(baseline_best_df,on=['Set size', 'Subject number', 'Stim. condition'],how='left')
        df_best_minus_baseline['Model name'] = df_best_minus_baseline['Model name_x']
        df_best_minus_baseline[f'LLH improvement'] = df_best_minus_baseline[f'Train LLH - spread over subjects_x'] - df_best_minus_baseline[f'Train LLH - spread over subjects_y']
        df_best_minus_baseline[f'BIC improvement'] = df_best_minus_baseline[f'Train BIC - spread over subjects_x'] - df_best_minus_baseline[f'Train BIC - spread over subjects_y']

        fig, axes = plt.subplots(1, 2, figsize = (4, 6))

        df_best_minus_baseline = df_best_minus_baseline[df_best_minus_baseline['Model name_x'].str.contains('stable')]

        if dataset_name == 'mcmaster2022_e2_dircue':
            drop_idx = df_best_minus_baseline[df_best_minus_baseline['Model name'].str.contains('cue')][df_best_minus_baseline['Subject number'] == 9][df_best_minus_baseline['Stim. condition']=='_medC'].index
            assert len(drop_idx) == 1
            df_best_minus_baseline = df_best_minus_baseline.drop(drop_idx[0], axis = 0)

        df_best_minus_baseline['Model name'] = df_best_minus_baseline['Model name'].map(lambda x: {
            'spike_and_slab_wrapped_stable': 'Simple',
            'cue_dim_wrapped_stable': 'Probe',
            'est_dim_wrapped_stable': 'Report',
            'full_wrapped_stable': 'Both'
        }.get(x, x))

        order = ['Simple', 'Report', 'Probe', 'Both']
        palette = [PALETTE.sns, PALETTE.est_dim, PALETTE.cue_dim, PALETTE.full]

        sns.stripplot(data = df_best_minus_baseline[df_best_minus_baseline['Subject number'] != 'Aggregated'], y=f'LLH improvement', x='Model name', s=10, alpha = 0.2, ax = axes[0], order = order, palette = palette)
        sns.stripplot(data = df_best_minus_baseline[df_best_minus_baseline['Subject number'] != 'Aggregated'], y=f'BIC improvement', x='Model name', s=10, alpha = 0.2, ax = axes[1], order = order, palette = palette)
        sns.stripplot(data = df_best_minus_baseline[df_best_minus_baseline['Subject number'] == 'Aggregated'], y=f'LLH improvement', x='Model name', s=15, marker = '^', ax = axes[0], order = order, palette = palette)
        sns.stripplot(data = df_best_minus_baseline[df_best_minus_baseline['Subject number'] == 'Aggregated'], y=f'BIC improvement', x='Model name', s=15, marker = '^', ax = axes[1], order = order, palette = palette)

        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[1].spines['left'].set_visible(False)
        axes[1].set_yticks([])
        axes[1].set_yticklabels([])
        axes[1].set_title(axes[1].get_ylabel().split(' ')[0])
        axes[1].set_xlabel('')
        axes[1].set_ylabel('')
        axes[0].set_title(axes[0].get_ylabel().split(' ')[0])
        axes[0].set_xlabel('')
        axes[0].set_ylabel('')

        axes[0].tick_params(axis='x', labelrotation=45, labelsize = 12)
        axes[1].tick_params(axis='x', labelrotation=45, labelsize = 12)
        axes[0].tick_params(axis='y', labelsize = 12)

        if 'dircue' in dataset_name:
            axes[0].set_yticks([0.0, 0.2])
        else:
            axes[0].set_yticks([0.0, 0.4])
        axes[0].set_ylabel('Nats', labelpad = -20, fontsize = 15)


        dots = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='Per subject')
        triangles = mlines.Line2D([], [], color='black', marker='^', linestyle='None', markersize=10, label='Aggregated')
        axes[1].legend(handles=[dots, triangles], fontsize = 10)

        y_min = min([ax.get_ylim()[0] for ax in axes.flatten()])
        y_max = max([ax.get_ylim()[1] for ax in axes.flatten()])
        [ax.set_ylim(y_min, y_max) for ax in axes.flatten()]

        plt.tight_layout()
        plt.savefig(os.path.join(figure_base, f'final_figure_llh_{dataset_name}.svg'), pad_inches=0, transparent=True, format = 'svg',)
        import pdb; pdb.set_trace(header = "Printed to " + os.path.join(figure_base, f'final_figure_llh_{dataset_name}.svg'))


    # elif dataset_name == 'schneegans2017_e2_cueOrientation_reportColor':
    elif 'schneegans2017' in dataset_name:
        
        fig, axes = plt.subplots(1, 1, figsize = (2, 6))

        baseline_best_df = df_best[df_best['Model name'] == BASELINE_MODEL_NAME_ALT]
        df_best_minus_baseline = df_best.merge(baseline_best_df,on=['Set size', 'Subject number', 'Stim. condition'],how='left')
        df_best_minus_baseline['Model name'] = df_best_minus_baseline['Model name_x']
        df_best_minus_baseline[f'LLH improvement'] = df_best_minus_baseline[f'Train LLH - spread over subjects_x'] - df_best_minus_baseline[f'Train LLH - spread over subjects_y']
        df_best_minus_baseline[f'BIC improvement'] = df_best_minus_baseline[f'Train BIC - spread over subjects_x'] - df_best_minus_baseline[f'Train BIC - spread over subjects_y']

        df_best_minus_baseline = df_best_minus_baseline[df_best_minus_baseline['Model name_x'].str.contains('stable')]

        df_best_minus_baseline['Model name'] = df_best_minus_baseline['Model name'].map(lambda x: {
            'spike_and_slab_wrapped_stable': 'Simple',
            'cue_dim_wrapped_stable': 'Probe',
            'est_dim_wrapped_stable': 'Report',
            'full_wrapped_stable': 'Both'
        }.get(x, x))

        order = ['Simple', 'Report', 'Probe', 'Both']
        palette = [PALETTE.sns, PALETTE.est_dim, PALETTE.cue_dim, PALETTE.full]
        
        sns.stripplot(data = df_best_minus_baseline[df_best_minus_baseline['Subject number'] == 'Aggregated'], y=f'LLH improvement', s = 15, x='Model name', marker = 'o', ax = axes, order = order, palette = palette)
        sns.stripplot(data = df_best_minus_baseline[df_best_minus_baseline['Subject number'] == 'Aggregated'], y=f'BIC improvement', s = 15, x='Model name', marker = '^', ax = axes, order = order, palette = palette)

        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.set_xlabel('')
        axes.set_yticks([0.0, 0.07])
        axes.set_ylabel('Nats', labelpad = -20, fontsize = 15)

        axes.set_yticks([0.0, 0.07])
        axes.tick_params(axis='x', labelrotation=45, labelsize = 15)
        axes.tick_params(axis='y', labelsize = 12)
        axes.set_ylim(-0.015, 0.075)

        dots = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='LLH')
        triangles = mlines.Line2D([], [], color='black', marker='^', linestyle='None', markersize=10, label='BIC')
        axes.legend(handles=[dots, triangles], fontsize = 12, loc = 'center')

        plt.tight_layout()
        plt.savefig(os.path.join(figure_base, f'final_figure_llh_{dataset_name}.svg'), pad_inches=0, transparent=True, format = 'svg',)
        import pdb; pdb.set_trace(header = "Printed to " + os.path.join(figure_base, f'final_figure_llh_{dataset_name}.svg'))

