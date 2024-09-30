import numpy as np
from os.path import join
from glob import glob
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, ttest_1samp


LOGGING_BASE = '/homes/pr450/repos/research_projects/error_modelling_torus/results_link/e_synthetic_likelihood_24_9_24'


DATASET_ALIASES_DICT = {
    "mcmaster2022_e1_oricue_cue_AR2": "McMaster 2022 - orientation cued, location recalled",
    "mcmaster2022_e2_dircue_medC": "McMaster 2022 - direction cued, location recalled",
}
MODEL_STRUCTURE_ALIASES_DICT = {
    'cue_dim_only_von_mises_and_uniform': '$f$(cue); vM + uniform',
    'est_dim_only_von_mises_and_uniform': '$f$(rec); vM + uniform',
    'spike_and_slab_von_mises_and_uniform': 'flat swap; vM + uniform',
    'cue_dim_only_wrapped_stable': '$f$(cue); wS',
    'est_dim_only_wrapped_stable': '$f$(rec); wS',
    'spike_and_slab_wrapped_stable': 'flat swap; wS',
}

DATASET_CODES = ['mcmaster2022_e1_oricue_cue_AR2', 'mcmaster2022_e2_dircue_medC']

if DATASET_CODES is None:
    dataset_codes = list(map(lambda x: x.split('/')[-1], glob(join(LOGGING_BASE, 'trained_on_data', '*'))))  # 'mcmaster2022_...' etc.
else:
    dataset_codes = DATASET_CODES


# fig, axes = plt.subplots(len(dataset_codes), 2, figsize = (15, 5*len(dataset_codes)))
fig = plt.figure(figsize = (15, 5*len(dataset_codes)))
gs = fig.add_gridspec(len(dataset_codes),3)


for dataset_index, dataset_code in enumerate(dataset_codes):

    data_axes = fig.add_subplot(gs[dataset_index, :2])

    dataset_alias = DATASET_ALIASES_DICT.get(dataset_code, dataset_code)
    data_axes.set_title(dataset_alias, fontsize = 20)

    model_structure_codes = list(map(lambda x: x.split('/')[-1], glob(join(LOGGING_BASE, 'trained_on_data', dataset_code, '*'))))  # 'cue_dim_o...' etc.

    y_offset = 0.0

    for model_structure_code in model_structure_codes:

        data_results_path = join(LOGGING_BASE, 'trained_on_data', dataset_code, model_structure_code, 'run_0', 'recent_losses.npy')
        synth_results_path = join(LOGGING_BASE, 'trained_on_synth', dataset_code, model_structure_code, 'run_0', 'recent_losses.npy')

        try:
            data_results = np.load(data_results_path, allow_pickle=True).item()['recent_train_llh']
            synth_results = np.load(synth_results_path, allow_pickle=True).item()['recent_train_llh']
            assert len(data_results) == len(synth_results) == 1, 'Not doing multiple set sizes for visualisation yet!'
        except Exception as e:
            print(e, dataset_code, model_structure_code)
            continue
        
        set_size = list(data_results.keys())[0]
        data_llh = data_results[set_size]
        synth_llhs = synth_results[set_size]

        y_offset += 3.0

        model_structure_alias = MODEL_STRUCTURE_ALIASES_DICT.get(model_structure_code, model_structure_code)
        color = data_axes.plot(data_llh, np.zeros_like(data_llh) + y_offset, marker = 'x', markersize = 10, label = model_structure_alias)[0].get_color()
        data_axes.scatter(synth_llhs, 0.25 * np.random.randn(*synth_llhs.shape) + y_offset, marker = 'o', c = color)
        
        
        if ttest_1samp(synth_llhs, data_llh).pvalue < 0.05:
            data_axes.plot(data_llh, np.zeros_like(data_llh) + y_offset, marker = '*', markersize = 10, color = 'black')


        # Also check for any cross-fits:
        for generating_model_structure_code in model_structure_codes:

            cross_synth_results_path = join(LOGGING_BASE, 'trained_on_synth_different_model', dataset_code, f"trained_on_{generating_model_structure_code}", model_structure_code, 'run_0', 'recent_losses.npy')

            try:
                cross_synth_results = np.load(cross_synth_results_path, allow_pickle=True).item()['recent_train_llh']
            except Exception as e:
                print(e, dataset_code, model_structure_code, generating_model_structure_code)
                continue

            set_size = list(cross_synth_results.keys())[0] # XXX: needs to be looped with more set sizes allowed!
            cross_synth_llhs = cross_synth_results[set_size]

            y_offset += 1.0
            generating_model_structure_alias = MODEL_STRUCTURE_ALIASES_DICT.get(generating_model_structure_code, generating_model_structure_code)
            data_axes.scatter(cross_synth_llhs, 0.25 * np.random.randn(*cross_synth_llhs.shape) + y_offset, marker = '$\u266B$', color = color, label = f"{model_structure_alias} fitted to {generating_model_structure_alias}", s = 100)




    data_axes.legend(fontsize = 10, loc='center left', bbox_to_anchor=(1, 0.5))
    data_axes.set_yticks([])



print('ALSO DO IN MODEL LLH')

figure_path = '/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/e_synthetic_loglikelihood/results/llh_comparison_figure.png'
fig.savefig(figure_path)
