from glob import glob
from purias_utils.util.arguments_yaml import ConfigNamepace

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import matplotlib as mpl

from boxplot_utils import plot_ttest_rel_significance_against_baseline_at_zero_pivot

dataset_names = None# ['bays2009']
do_kurtosis = False
do_fitted_kurtosis = False

config_base_path = "/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data"
all_yaml_paths = glob(os.path.join(config_base_path, "*.yaml"))
all_yamls = [ConfigNamepace.from_yaml_path(yaml_path) for yaml_path in all_yaml_paths]

if dataset_names is None:
    dataset_names = list(set([x.dataset_name for x in all_yamls]))
else:
    dataset_names = dataset_names

for dataset_name in dataset_names:

    results_txt = f'/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/figures/source_tables/model_comparison_and_kurtosis_{dataset_name}.txt'
    results_df = pd.read_csv(results_txt, sep = ',')

    base_line_model_name = 'spike_and_slab_von_mises_and_uniform'
    model_class_order = sorted(results_df.model_class.unique())
    model_class_order.remove(base_line_model_name)

    model_comparison_figure_path = f'/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/figures/model_comparison/{dataset_name}'

    Path(model_comparison_figure_path).mkdir(parents = True, exist_ok = True)

    for N in results_df.N.unique():

        set_size_results_df = results_df[results_df.N == N]

        assert len(set_size_results_df.train_sizes.unique()) == 1
        assert len(set_size_results_df.test_sizes.unique()) == 1

        fig, [ax_train, ax_test] = plt.subplots(1, 2, figsize = (14, 10))

        ax_train.set_title(f'Train LLH (averaged)\nimprovement over {base_line_model_name}')
        ax_test.set_title(f'Test LLH (averaged)\nimprovement over {base_line_model_name}')

        train_pivot = set_size_results_df.pivot(index = 'model_class', columns = 'training_set_seed', values = 'train_llh')# .dropna(axis = 1)
        train_pivot = train_pivot / set_size_results_df.train_sizes.iloc[0]
        train_pivot_difference = train_pivot - train_pivot.loc[base_line_model_name]
        train_results_table = train_pivot_difference.drop(base_line_model_name, axis = 0).transpose()
        sns.boxplot(data=train_results_table, ax = ax_train, boxprops={'alpha': 0.4}, order = model_class_order)
        sns.stripplot(data=train_results_table, ax = ax_train)
        plot_ttest_rel_significance_against_baseline_at_zero_pivot(ax_train, train_results_table, 0.001)

        test_pivot = set_size_results_df.pivot(index = 'model_class', columns = 'training_set_seed', values = 'test_llh').dropna(axis = 1)
        test_pivot = test_pivot / set_size_results_df.test_sizes.iloc[0]
        test_pivot_difference = test_pivot - test_pivot.loc[base_line_model_name]
        test_results_table = test_pivot_difference.drop(base_line_model_name, axis = 0).transpose()
        sns.boxplot(data=test_results_table, ax = ax_test, boxprops={'alpha': 0.4}, order = model_class_order)
        sns.stripplot(data=test_results_table, ax = ax_test)
        plot_ttest_rel_significance_against_baseline_at_zero_pivot(ax_test, test_results_table, 0.001)

        for ax in [ax_train, ax_test]:
            ax.tick_params(labelrotation=10)

        fig.savefig(os.path.join(model_comparison_figure_path, f'model_comparison_results_{dataset_name}_N{N}.png'))


    if do_kurtosis:

        # Based on Bays2016 Figure 7d: split by model, and plot the following against set size for each model class:
            # Figure 1: 'residual_kurtosis_fitted' - 'raw_kurtosis_fitted'
            # Figure 1: 'residual_kurtosis_particle' - 'raw_kurtosis_particle'
            # Figure 2: 'emissions_kurtosis' - 'raw_kurtosis_fitted'
            # Figure 2: 'emissions_kurtosis' - 'raw_kurtosis_particle'
        # with minus signs being for the same runs

        if do_fitted_kurtosis:
            fig_kurtosis, axes_kurtosis = plt.subplots(2, 1, figsize = (7, 14))
        else:
            fig_kurtosis, axes_kurtosis = plt.subplots(1, 1, figsize = (7, 7))

        kurtosis_results_df = results_df[results_df.model_class.str.contains('wrapped_stable')]

        full_model_class_order = sorted(kurtosis_results_df.model_class.unique())

        difference_pairs = [
            ('residual_kurtosis_fitted', 'raw_kurtosis_fitted'),
            ('residual_kurtosis_particle', 'raw_kurtosis_particle'),
            ('emissions_kurtosis', 'raw_kurtosis_fitted'),
            ('emissions_kurtosis', 'raw_kurtosis_particle'),
        ]

        for post_col, pre_col in difference_pairs:

            if 'fitted' in pre_col and not do_fitted_kurtosis:
                continue
            elif do_fitted_kurtosis:
                raise Exception

            new_key = f'{post_col} - {pre_col}'
            kurtosis_results_df[new_key] = kurtosis_results_df.apply(lambda x: x[post_col] - x[pre_col], axis=1)
            
            bp = sns.boxplot(x = 'N', y=new_key, hue='model_class', hue_order=full_model_class_order, data = kurtosis_results_df, ax = axes_kurtosis)

        fig_kurtosis.savefig(os.path.join(model_comparison_figure_path, f'kurtosis_comparison_results_{dataset_name}.png'))
