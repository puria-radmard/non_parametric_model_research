import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

from non_parametric_model.commands.fc_single_subject_summaries.final_figures import get_exp_surface_from_path

from non_parametric_model.commands.fc_single_subject_summaries import PALETTE

dataset_names = ['schneegans2017_e2_cueOrientation_reportColor', 'schneegans2017_e2_cueColor_reportOrientation']

results_base_path = '/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_summaries_7_10_24/schneegans2017_e2_cueOrientation_reportColor'

models = {
    # 'full_wrapped_stable': (9, False),
    '/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_summaries_7_10_24/schneegans2017_e2_cueOrientation_reportColor/cue_dim_wrapped_stable_0': (9, True),
    '/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_summaries_7_10_24/schneegans2017_e2_cueOrientation_reportColor/est_dim_wrapped_stable_0': (9, True),
    '/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_summaries_7_10_24/schneegans2017_e2_cueColor_reportOrientation/cue_dim_wrapped_stable_0': (0, True),
    '/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_summaries_7_10_24/schneegans2017_e2_cueColor_reportOrientation/est_dim_wrapped_stable_0': (-3, True),
}

MIN_SEP = 10/180*np.pi
SET_SIZE = 6

surface_infos = {k: get_exp_surface_from_path(k,v[0],True,MIN_SEP,SET_SIZE,v[1]) for k, v in models.items()}



fig = plt.figure(constrained_layout = False, figsize = (5, 5))

# fig = plt.figure(constrained_layout = False, figsize = (17, 17))
# fig_surfaces_spec = gridspec.GridSpec(7, 7, fig)
# axes_full_surface = fig.add_subplot(fig_surfaces_spec[:3,:3])
# _, _, full_surface, _ = surface_infos['full_wrapped_stable']
# edge_size = int(full_surface.shape[0]**0.5)
# axes_full_surface.imshow(full_surface.reshape(edge_size, edge_size).T, extent = [MIN_SEP, np.pi, MIN_SEP, np.pi])
# axes_full_surface.set_xticks([MIN_SEP, np.pi])
# axes_full_surface.set_yticks([MIN_SEP, np.pi])
# axes_full_surface.set_xticklabels(['$10^\circ$', '$180^\circ$'], fontsize = 20)
# axes_full_surface.set_yticklabels(['$10^\circ$', '$180^\circ$'], fontsize = 20)
# axes_full_surface.set_xlabel('Orientation (probe)', labelpad=-20, fontsize = 20)
# axes_full_surface.set_ylabel('Colour (report)', labelpad=-40, fontsize = 20)


# axes_1D_surfaces = fig.add_subplot(fig_surfaces_spec[4:,4:])
axes_1D_surfaces = fig.add_subplot(111)
_, oned_grid, est_surface_colour, _ = surface_infos['/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_summaries_7_10_24/schneegans2017_e2_cueOrientation_reportColor/est_dim_wrapped_stable_0']
_, _, cue_surface_orientation, _ = surface_infos['/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_summaries_7_10_24/schneegans2017_e2_cueOrientation_reportColor/cue_dim_wrapped_stable_0']
_, _, est_surface_orientation, _ = surface_infos['/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_summaries_7_10_24/schneegans2017_e2_cueColor_reportOrientation/est_dim_wrapped_stable_0']
_, _, cue_surface_colour, _ = surface_infos['/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_summaries_7_10_24/schneegans2017_e2_cueColor_reportOrientation/cue_dim_wrapped_stable_0']
axes_1D_surfaces.spines['top'].set_visible(False)
axes_1D_surfaces.spines['right'].set_visible(False)
[PALETTE.sns, PALETTE.est_dim, PALETTE.cue_dim, PALETTE.full]
axes_1D_surfaces.plot(oned_grid, est_surface_orientation, linewidth = 5, linestyle = '-', color = PALETTE.est_dim)
axes_1D_surfaces.plot(oned_grid, cue_surface_orientation, linewidth = 5, linestyle = '-', color = PALETTE.cue_dim)
axes_1D_surfaces.plot(oned_grid, est_surface_colour, linewidth = 5, linestyle = '--', color = PALETTE.est_dim)
axes_1D_surfaces.plot(oned_grid, cue_surface_colour, linewidth = 5, linestyle = '--', color = PALETTE.cue_dim)
axes_1D_surfaces.set_xticks([MIN_SEP, np.pi])
axes_1D_surfaces.set_xticklabels(['$10^\circ$', '$180^\circ$'], fontsize = 15)
axes_1D_surfaces.set_xlabel('Distractor feature value', labelpad=-15, fontsize = 15)
axes_1D_surfaces.set_ylabel('$\langle e^f \\rangle_q$', fontsize = 15)
axes_1D_surfaces.legend(fontsize = 20)


label_probed = mlines.Line2D([], [], color=PALETTE.cue_dim, marker=None, linestyle='-', label='As probed')
label_reported = mlines.Line2D([], [], color=PALETTE.est_dim, marker=None, linestyle='-', label='As reported')
label_orientation = mlines.Line2D([], [], color='black', marker=None, linestyle='-', label='Orientation')
label_colour = mlines.Line2D([], [], color='black', marker=None, linestyle='--', label='Colour')
axes_1D_surfaces.legend(handles=[label_orientation, label_colour, label_probed, label_reported], fontsize = 10)


fig.savefig(
    '/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/final_figure_3.svg',
    # bbox_inches='tight', 
    pad_inches=0, transparent=True, format = 'svg'
)
