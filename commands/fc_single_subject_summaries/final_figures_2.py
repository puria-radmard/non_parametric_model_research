
import os, torch
import numpy as np

import matplotlib.pyplot as plt

from purias_utils.util.arguments_yaml import ConfigNamepace

from purias_utils.error_modelling_torus.non_parametric_error_model.setup_utils import setup_model_whole
from purias_utils.error_modelling_torus.data_utils.loading_utils import load_experimental_data

from non_parametric_model.commands.fc_single_subject_summaries import PALETTE


SET_SIZE = 6
CHOSEN_MODEL = 10
MIN_SEP = np.pi / 5

cue_only_fit_path = '/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_summaries_7_10_24/mcmaster2022_e3_rand/cue_dim_wrapped_stable_0'


result_args = ConfigNamepace.from_yaml_path(os.path.join(cue_only_fit_path, 'args.yaml'))
result_args.update(ConfigNamepace({'all_set_sizes': [SET_SIZE], 'trainable_kernel_delta': False, 'num_variational_samples': 4096, 'num_importance_sampling_samples': 4096}))
result_args.dict['resume_path'] = cue_only_fit_path
swap_model, D, delta_dimensions = setup_model_whole(**result_args.dict, all_min_seps = None, device = 'cuda')
swap_model.reduce_to_single_model(CHOSEN_MODEL)
swap_model.cuda()

grid_inference_info = swap_model.inference_on_grid(set_size=SET_SIZE, grid_count=360)
all_grid_points = grid_inference_info['all_grid_points'][0,:,1,0]
include_mask = (all_grid_points >= MIN_SEP)
masked_grid_points = all_grid_points[include_mask]
mean_surface = grid_inference_info['mean_surface'][0][include_mask]
std_surface = grid_inference_info['std_surface'][0][include_mask]
exp_surface = np.exp(mean_surface + 0.5 * std_surface)
exp_std_surface = (mean_surface**2) * (np.exp(std_surface) - 1.)


fig = plt.figure(figsize = (5, 5))
axes_surface = fig.add_subplot(111)
axes_surface.spines['top'].set_visible(False)
axes_surface.spines['right'].set_visible(False)
axes_surface.plot(masked_grid_points, exp_surface, color = PALETTE.cue_dim, linewidth = 7)
axes_surface.set_xticks([MIN_SEP, np.pi])
axes_surface.set_xticklabels(['$\pi/5$', '$\pi$'])
axes_surface.set_xlabel('Grating location (probe)', labelpad=-15, fontsize = 15)
axes_surface.set_ylabel('$\langle e^f \\rangle_q$', labelpad=-10, fontsize = 15)
axes_surface.set_ylim([-0.05, 1.05])
axes_surface.set_yticks([0.0, 1.0])
# axes_surface.plot([MIN_SEP, np.pi], [np.e, np.e])

fig.savefig(
    '/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/final_figure_2.svg',
    bbox_inches='tight', pad_inches=0, transparent=True, format = 'svg'
)
