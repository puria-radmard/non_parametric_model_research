import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


# Extract LLHs per dataset
recovery = np.load('/homes/pr450/repos/research_projects/error_modelling_torus/results_link/h_hierarchical_tests_19_11_11/model_recovery/mcmaster2022_e1_oricue_cue_AR2/cue_dim_wrapped_stable_0/recent_losses.npy', allow_pickle = True).item()
inmodel = np.load('/homes/pr450/repos/research_projects/error_modelling_torus/results_link/h_hierarchical_tests_19_11_11/synthetic_data/hierarchical_synthetic_data_mcmaster2022_e1_oricue_cue_AR2_cue_dim_wrapped_stable_gaussian/synthetic_data_hierarchical.npy', allow_pickle = True).item()

recovered_llhs = np.stack([recovery['recent_naive_log_likelihoods'][6][si] for si in range(10)], 1)
inmodel_llhs = np.stack([inmodel['inmodel_loglikelihood'][6][:,si * 96: (si + 1) * 96] for si in range(10)], 1)

diff_llhs = (recovered_llhs - inmodel_llhs).mean(-1)

for ridx, rep_diff_llhs in enumerate(diff_llhs):
    plt.scatter(ridx + np.random.rand(*rep_diff_llhs.shape) * 0.1, rep_diff_llhs, color = 'blue', alpha = 0.4)
    ptest = ttest_rel(rep_diff_llhs, np.zeros_like(rep_diff_llhs))
    if ptest.pvalue < 0.05:
        plt.scatter(ridx, 0.1, marker = '*', color = 'black')

plt.plot([-0.1, ridx + 0.1], [0., 0.], color = 'black', linestyle = '--')
plt.xlabel('Synthetic datasets')
plt.xticks(range(ridx + 1))
plt.suptitle('recovered_llhs - inmodel_llhs across synthetic subjects')
plt.savefig('/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/h_hierarchical_tests/llh_comparison_figures/recovery_llhs')


