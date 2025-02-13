import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
import math

from non_parametric_model.commands.fc_single_subject_summaries import PALETTE
from non_parametric_model.commands.fc_single_subject_summaries.bic_comparison import extract_results, generate_df_cats


results_base = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link/g_underfitting_checks"
figure_base = "/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/g_underfitting_checks/llh_comparison_figures"
fit_to_subdir_indiv = "fit_to_full_wrapped_stable_reduced"
fit_to_subdir_full = "fit_to_full_wrapped_stable_notreduced"

dataset_name = 'mcmaster2022_e2_dircue'
subject_indices = range(10)
set_sizes = [4]
stim_conditions = ['_lowC', '_medC', '_highC']



model_names = [
    'spike_and_slab_wrapped_stable',
    'est_dim_wrapped_stable',
    'cue_dim_wrapped_stable',
    'full_wrapped_stable',
]

BASELINE_MODEL_NAME = model_names[0]

FIG_HEIGHT = 3
FIG_ASPECT = 1.2
SHOWFLIERS=False

def generate_df_cats_for_synthetic(set_size, results_dict, num_params, subject_code, model_code, scond):
    "Unike original case, every repeat is different here so it matters!"

    try:
        average_training_llh = results_dict['recent_naive_log_likelihoods'][set_size].mean(-1).cpu().numpy()
    except KeyError:
        print(log_base)
        return None, None

    num_data = results_dict['recent_naive_log_likelihoods'][set_size].shape[1]

    bic_adjustment = num_params * math.log(num_data) / num_data

    df_cat_reps = pd.DataFrame(
        {
            'Train LLH - spread over subjects': average_training_llh,
            'Train BIC - spread over subjects': average_training_llh - bic_adjustment,
        }
    )
    df_cat_reps['Model name'] = model_code
    df_cat_reps['Subject number'] = subject_code
    df_cat_reps['Set size'] = set_size
    df_cat_reps['Stim. condition'] = scond
    df_cat_reps['num_params'] = num_params
    df_cat_reps['num_data'] = num_data
    df_cat_reps['Train BIC_adjustment - spread over subjects'] = bic_adjustment         # XXX These "spread over subjects" headers are a little white lie! Shoudl be subjects and iterations
    df_cat_reps['Repeat number'] = range(len(df_cat_reps))

    return df_cat_reps
   


if __name__ == '__main__':

    assert (len(set_sizes) == 1 or len(stim_conditions) == 1)
    # if len(stim_conditions) == 1:
    #     assert stim_conditions[0] == ''

    df_best = pd.DataFrame()

    for stim_condition in stim_conditions:

        for model_name in model_names:

            ######## AGGREGATED
            log_base = os.path.join(results_base, f"{dataset_name}{stim_condition}", fit_to_subdir_full, f"{model_name}_0")
            try:
                results, num_params = extract_results(log_base)
            except (FileNotFoundError, AssertionError) as e:
                print(f'MISSING ({e})', log_base)
                continue

            for set_size in set_sizes:
                df_cat_best = generate_df_cats_for_synthetic(set_size, results, num_params, 'Aggregated', model_name, stim_condition)
                df_best = pd.concat([df_best, df_cat_best], ignore_index=True)

            ######## NOT AGGREGATED
            for subject_index in subject_indices:

                log_base = os.path.join(results_base, f"{dataset_name}{stim_condition}", fit_to_subdir_indiv, f"{model_name}_{subject_index}")
                try:
                    results, num_params = extract_results(log_base)
                except (FileNotFoundError, AssertionError) as e:
                    print(f'MISSING ({e})', log_base)
                    continue

                for set_size in set_sizes:
                    df_cat_best = generate_df_cats_for_synthetic(set_size, results, num_params, subject_index, model_name, stim_condition)
                    df_best = pd.concat([df_best, df_cat_best], ignore_index=True)


    if 'mcmaster' in dataset_name:

        baseline_best_df = df_best[df_best['Model name'] == BASELINE_MODEL_NAME]
        df_best_minus_baseline = df_best.merge(baseline_best_df,on=['Set size', 'Subject number', 'Stim. condition', 'Repeat number'],how='left')
        df_best_minus_baseline['Model name'] = df_best_minus_baseline['Model name_x']
        df_best_minus_baseline[f'LLH improvement'] = df_best_minus_baseline[f'Train LLH - spread over subjects_x'] - df_best_minus_baseline[f'Train LLH - spread over subjects_y']
        df_best_minus_baseline[f'BIC improvement'] = df_best_minus_baseline[f'Train BIC - spread over subjects_x'] - df_best_minus_baseline[f'Train BIC - spread over subjects_y']

        fig, axes = plt.subplots(1, 2, figsize = (4, 6))

        df_best_minus_baseline = df_best_minus_baseline[df_best_minus_baseline['Model name_x'].str.contains('stable')]

        #if dataset_name == 'mcmaster2022_e2_dircue':
        #    drop_idx = df_best_minus_baseline[df_best_minus_baseline['Model name'].str.contains('cue')][df_best_minus_baseline['Subject number'] == 9][df_best_minus_baseline['Stim. condition']=='_medC'].index
        #    assert len(drop_idx) == 1
        #    df_best_minus_baseline = df_best_minus_baseline.drop(drop_idx[0], axis = 0)

        df_best_minus_baseline['Model name'] = df_best_minus_baseline['Model name'].map(lambda x: {
            'spike_and_slab_wrapped_stable': 'Simple',
            'est_dim_wrapped_stable': 'Report',
            'cue_dim_wrapped_stable': 'Probe',
            'full_wrapped_stable': 'Both'
        }.get(x, x))

        order = ['Simple', 'Report', 'Probe', 'Both']
        palette = [PALETTE.sns, PALETTE.est_dim, PALETTE.cue_dim, PALETTE.full]

        sns.stripplot(data = df_best_minus_baseline[df_best_minus_baseline['Subject number'] != 'Aggregated'], y=f'LLH improvement', x='Model name', s=10, alpha = 0.1, ax = axes[0], order = order, palette = palette)
        sns.stripplot(data = df_best_minus_baseline[df_best_minus_baseline['Subject number'] != 'Aggregated'], y=f'BIC improvement', x='Model name', s=10, alpha = 0.1, ax = axes[1], order = order, palette = palette)
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

        axes[0].set_ylabel('Nats', labelpad = -20, fontsize = 15)


        dots = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='Per subject')
        triangles = mlines.Line2D([], [], color='black', marker='^', linestyle='None', markersize=10, label='Aggregated')
        axes[1].legend(handles=[dots, triangles], fontsize = 10)

        y_min = -0.1 # min([ax.get_ylim()[0] for ax in axes.flatten()])
        y_max = +0.1 # max([ax.get_ylim()[1] for ax in axes.flatten()])
        [ax.set_ylim(y_min, y_max) for ax in axes.flatten()]

        plt.tight_layout()
        # plt.savefig(os.path.join(figure_base, f'final_figure_llh_{dataset_name}.svg'), pad_inches=0, transparent=True, format = 'svg',)
        plt.savefig(os.path.join(figure_base, f'final_figure_llh_{dataset_name}.png'), pad_inches=0, transparent=False, format = 'png',)


        import pdb; pdb.set_trace()

        pass

