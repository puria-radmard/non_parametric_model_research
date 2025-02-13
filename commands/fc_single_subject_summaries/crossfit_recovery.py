# Positive result would be that BIC lets discriminate full fitted to est/cue for both 
    # This is cool because on real data BIC(full) > BIC(est or cue) only for the aggregated case, not for subject-seperated case
    # So if BIC discriminates at both levels here, then the original result is a happy one, i.e. in large data then the extra complexity of the full model is 'justified'


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from non_parametric_model.commands.fc_single_subject_summaries.bic_comparison import extract_results, generate_df_cats


model_names = [
    # 'spike_and_slab_wrapped_stable',
    'cue_dim_wrapped_stable',
    'est_dim_wrapped_stable',
    'full_wrapped_stable',
]



# dataset_name = 'mcmaster2022_e1_oricue'
# num_subjects = 10
# set_sizes = [6]
# stim_conditions = ['_cue_AR1', '_cue_AR2', '_cue_AR3']

dataset_name = 'mcmaster2022_e2_dircue'
num_subjects = 10
set_sizes = [4]
stim_conditions = ['_lowC', '_medC', '_highC']



results_base_single_subject = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_single_subject_crossfits_8_10_24"
results_base_aggregated_subject = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_crossfits_8_10_24"
figure_base = "/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/figures"


metrics = ['BIC', 'LLH']

figsize = 5
fig_aggs_dict, axes_aggs_dict = {}, {}
for metric in metrics:
    fig_agg, axes_agg = plt.subplots(len(model_names), len(stim_conditions), figsize = (len(stim_conditions) * figsize, len(model_names) * figsize))
    fig_aggs_dict[metric] = fig_agg
    axes_aggs_dict[metric] = axes_agg
del fig_agg, axes_agg


for ax_row, stim_condition in enumerate(stim_conditions):

    for ax in axes_aggs_dict.values():
        ax[0, ax_row].set_title(stim_condition)

    for ax_col, generating_model_name in enumerate(model_names):

        aggregated_df_synths = pd.DataFrame()

        for fitting_model_name in model_names:
            
            ######## AGGREGATED
            log_base = os.path.join(results_base_aggregated_subject, f"{dataset_name}{stim_condition}", f"fit_to_{generating_model_name}", f"{fitting_model_name}_0")
            results, num_params = extract_results(log_base)

            for set_size in set_sizes:
                
                print(log_base)
                df_cat_synths, _ = generate_df_cats(set_size, results, num_params, 'Aggregated', fitting_model_name, stim_condition)
                df_cat_synths['synth_repeat_num'] = df_cat_synths.index


                aggregated_df_synths = pd.concat([aggregated_df_synths, df_cat_synths], ignore_index=True)


        for metric in ['BIC', 'LLH']:

            true_model_values = aggregated_df_synths[aggregated_df_synths['Model name'] == generating_model_name]
            df_minus_true_model = aggregated_df_synths.merge(true_model_values,on=['Set size', 'Subject number', 'Stim. condition', 'synth_repeat_num'],how='left')
            df_minus_true_model[f'Train {metric} - spread over training repeats'] = df_minus_true_model[f'Train {metric} - spread over training repeats_x'] - df_minus_true_model[f'Train {metric} - spread over training repeats_y']
            df_minus_true_model['Fitting model'] = df_minus_true_model['Model name_x']
            
            sns.boxplot(data = df_minus_true_model, x = f'Train {metric} - spread over training repeats', y = 'Fitting model', ax = axes_aggs_dict[metric][ax_col, ax_row])

    print()

    for mn, fig in fig_aggs_dict.items():
        fig.tight_layout()
        save_path = f'non_parametric_model/commands/fc_single_subject_summaries/figures/synthetic_crossfit_{mn.lower()}_over_synthgen_repeats_{dataset_name}.png'
        print(save_path)
        fig.savefig(save_path)


            # # ######## NOT AGGREGATED
            # # for subject_index in range(num_subjects):
                
            # #     log_base = os.path.join(results_base_single_subject, f"{dataset_name}{stim_condition}", f"fit_to_{generating_model_name}", f"{fitting_model_name}_{subject_index}")
            # #     results, num_params = extract_results(log_base)


            # #     for set_size in set_sizes:
            # #         df_cat_synths, _ = generate_df_cats(set_size, results, num_params, subject_index, fitting_model_name, stim_condition)




