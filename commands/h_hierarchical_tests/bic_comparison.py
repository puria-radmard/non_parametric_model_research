import os
import math
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from purias_utils.util.arguments_yaml import ConfigNamepace

from non_parametric_model.commands.fc_single_subject_summaries.bic_comparison import extract_results, generate_df_cats


results_base_single_subject = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_single_subject_summaries_7_10_24"
results_base_aggregated_subject = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_summaries_7_10_24"
results_base_hierarchical_subject = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link/h_hierarchical_tests_19_11_11/trained_on_data"
figure_base = "/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/h_hierarchical_tests/llh_comparison_figures"

all_model_names = [
    'spike_and_slab_von_mises_and_uniform',
    'spike_and_slab_wrapped_stable',
    'cue_dim_von_mises_and_uniform',
    'cue_dim_wrapped_stable',
    'est_dim_von_mises_and_uniform',
    'est_dim_wrapped_stable',
    'full_von_mises_and_uniform',
    'full_wrapped_stable',
]

BASELINE_MODEL_NAME = all_model_names[1]

model_names_agg = list(filter(lambda x: 'von_mises' not in x, all_model_names))
model_names_hier = ['full_wrapped_stable']# list(filter(lambda x: 'full' in x, all_model_names))

# dataset_name = 'mcmaster2022_e1_oricue'
# # subject_indices = [1, 2, 4, 5]
# subject_indices = range(10)
# set_size = 6
# stim_condition = '_cue_AR2'


dataset_name = 'mcmaster2022_e2_dircue'
# subject_indices = [0, 2, 5, 9]
subject_indices = range(10)
set_size = 4
stim_condition = '_medC'



def extract_results_hierarchical(log_base_addr):
    results_path = os.path.join(log_base_addr, "recent_losses.npy")
    config_path = os.path.join(log_base_addr, "args.yaml")
    model_path = os.path.join(log_base_addr, "swap_model.mdl")
    log_path = os.path.join(log_base_addr, "epoch_log_train.csv")

    recreated_args = ConfigNamepace.from_yaml_path(config_path)
    assert not (recreated_args.shared_emission_distribution or recreated_args.shared_swap_function), "BIC calc cannot work with sharing across set sizes"
    assert recreated_args.M_test_per_set_size == 0, "Calculation for no test set case!"

    model_params = torch.load(model_path)
    submodel_keys = set([k.removeprefix('generative_model.submodel_generative_models.')[0] for k in model_params.keys() if k.startswith('generative_model.submodel_generative_models')])
    param_counts = [v.numel() / recreated_args.num_models for k, v in model_params.items() if k.split('.')[0] == 'generative_model']
    num_params_per_submodel = sum(param_counts) / len(submodel_keys)

    results = np.load(results_path, allow_pickle=True).item()

    num_lines = sum(1 for _ in open(log_path))
    assert num_lines > 10, "Log too short, run clearly failed!"

    return results, num_params_per_submodel


def generate_df_cats_hierarchical(set_size, results_dict, num_params_per_submodel, submodel_keys, model_code, scond):

    average_training_llhs = {   # average over data of best case, which is the only case! (num_models = 1)
        smk: results_dict['recent_naive_log_likelihoods'][set_size].get(smk, torch.tensor([[float('nan')]])).mean(-1)
        for smk in submodel_keys
    }
    num_datas = {
        smk: results_dict['recent_naive_log_likelihoods'][set_size].get(smk, torch.tensor([[float('nan')]])).shape[1]
        for smk in submodel_keys
    }
    bic_adjustments = {
        smk: num_params_per_submodel * math.log(num_datas[smk]) / num_datas[smk]
        for smk in submodel_keys
    }

    to_list = lambda d: [d[smk] for smk in submodel_keys]
    to_list_subtract = lambda d1, d2: [d1[smk] - d2[smk] for smk in submodel_keys]

    assert all([average_training_llhs[k].shape == (1,) for k in submodel_keys]), "Cannot have num models > 1 for hierarchical model for now!"
    average_training_llhs = {k: v[0] for k, v in average_training_llhs.items()}

    df_cat = pd.DataFrame(
        {
            'Train LLH - spread over subjects': to_list(average_training_llhs),
            'Train BIC - spread over subjects': to_list_subtract(average_training_llhs, bic_adjustments),
            'Subject number': submodel_keys,                 # XXX: this is assuming hierarchy is just for subject data
            'num_data': to_list(num_datas),
            'num_params': [num_params_per_submodel for _ in num_datas],
            'Train BIC_adjustment - spread over subjects': to_list(bic_adjustments),
        }
    )

    df_cat['Stim. condition'] = scond
    df_cat['Set size'] = set_size
    df_cat['Model name'] = model_code + ' hierarchical'

    return df_cat



df_best = pd.DataFrame()



######## AGGREGATED NON HIERARCHICAL
# for model_name in model_names_agg:
#     log_base = os.path.join(results_base_aggregated_subject, f"{dataset_name}{stim_condition}", f"{model_name}_0")
#     results, num_params = extract_results(log_base)
#     print(num_params)

#     _, df_cat_best = generate_df_cats(set_size, results, num_params, 'Aggregated', model_name, stim_condition)
#     df_best = pd.concat([df_best, df_cat_best], ignore_index=True)


######## SUBJECT SEPERATED
for model_name in all_model_names:
    for subject_index in subject_indices:
        log_base = os.path.join(results_base_single_subject, f"{dataset_name}{stim_condition}", f"{model_name}_{subject_index}")

        try:
            results, num_params = extract_results(log_base)
        except (FileNotFoundError, AssertionError) as e:
            print(f'MISSING ({e})', log_base)
            continue

        _, df_cat_best = generate_df_cats(set_size, results, num_params, subject_index, model_name, stim_condition)
        df_best = pd.concat([df_best, df_cat_best], ignore_index=True)



######## HIERARCHICAL
num_params_per_submodel = None
for model_name in model_names_hier:
    log_base = os.path.join(results_base_hierarchical_subject, f"{dataset_name}{stim_condition}", f"{model_name}_0")
    results, num_params_per_submodel = extract_results_hierarchical(log_base)

    df_cat_best = generate_df_cats_hierarchical(set_size, results, num_params_per_submodel, subject_indices, model_name, stim_condition)
    df_best = pd.concat([df_best, df_cat_best], ignore_index=True)



######## HIERARCHICAL WITH SUBJECTS REMOVED
num_params_per_submodel = None
for model_name in model_names_hier:
    log_base = os.path.join(results_base_hierarchical_subject, f"{dataset_name}{stim_condition}_drop_bad_subjects", f"{model_name}_0")
    results, num_params_per_submodel = extract_results_hierarchical(log_base)

    df_cat_best = generate_df_cats_hierarchical(set_size, results, num_params_per_submodel, subject_indices, model_name + ' drop bad subjects', stim_condition)
    df_best = pd.concat([df_best, df_cat_best], ignore_index=True)




baseline_best_df = df_best[df_best['Model name'] == BASELINE_MODEL_NAME]
df_best_minus_baseline = df_best.merge(baseline_best_df,on=['Set size', 'Subject number', 'Stim. condition'],how='left')
df_best_minus_baseline['Model name'] = df_best_minus_baseline['Model name_x']
for stat_name in ['LLH', 'BIC', 'BIC_adjustment']:
    df_best_minus_baseline[f'Train {stat_name} - spread over subjects'] = df_best_minus_baseline[f'Train {stat_name} - spread over subjects_x'] - df_best_minus_baseline[f'Train {stat_name} - spread over subjects_y']


num_rows = 2
num_cols = len(model_names_hier)
fig, axes = plt.subplots(2, num_cols, figsize = (5 * num_cols, 5 * num_rows), squeeze = False)

def finalse_axes(ax):
    x_max = np.max(np.abs(ax.get_xlim()))
    y_max = np.max(np.abs(ax.get_ylim()))
    xy_max = max(x_max, y_max)
    ax.plot([-xy_max, xy_max], [-xy_max, xy_max], color = 'black', linestyle = '--')
    ax.plot([0, 0], [-xy_max, xy_max], color = 'black', linestyle = '--')
    ax.plot([-xy_max, xy_max], [0, 0], color = 'black', linestyle = '--')
    ax.set_ylim(ymin=-xy_max, ymax=xy_max)
    ax.set_xlim(xmin=-xy_max, xmax=xy_max)
    


for ax_col, comparable_model in zip(axes.T, model_names_hier):

    flat_version_df = df_best_minus_baseline[df_best_minus_baseline['Model name'] == comparable_model]
    hierarchical_version_df = df_best_minus_baseline[df_best_minus_baseline['Model name'] == comparable_model + ' hierarchical']
    hierarchical_version_df_dropped = df_best_minus_baseline[df_best_minus_baseline['Model name'] == comparable_model + ' drop bad subjects hierarchical']
    
    flat_vs_hier_df = flat_version_df.merge(hierarchical_version_df,on=['Set size', 'Subject number', 'Stim. condition'],how='left')
    for stat_name in ['LLH', 'BIC']:
        ax_col[0].scatter(flat_vs_hier_df[f'Train {stat_name} - spread over subjects_x'], flat_vs_hier_df[f'Train {stat_name} - spread over subjects_y'], label = stat_name)
        ax_col[0].set_xlabel('Flat')
        ax_col[0].set_ylabel('Hier')
    finalse_axes(ax_col[0])
    ax_col[0].legend()
    ax_col[0].set_title(comparable_model)

    flat_vs_hier_dropped_df = flat_version_df.merge(hierarchical_version_df_dropped,on=['Set size', 'Subject number', 'Stim. condition'],how='left')
    for stat_name in ['LLH', 'BIC']:
        ax_col[1].scatter(flat_vs_hier_dropped_df[f'Train {stat_name} - spread over subjects_x'], flat_vs_hier_dropped_df[f'Train {stat_name} - spread over subjects_y'], label = stat_name)
        ax_col[1].set_xlabel('Flat')
        ax_col[1].set_ylabel('Hier dropped')
    finalse_axes(ax_col[1])
    ax_col[1].legend()
    ax_col[1].set_title(comparable_model + ' drop bad subjects hierarchical')
    

fig.suptitle(f'Improvement over {BASELINE_MODEL_NAME}')
fig.savefig(os.path.join(figure_base, f'llh_{dataset_name}.png'))
