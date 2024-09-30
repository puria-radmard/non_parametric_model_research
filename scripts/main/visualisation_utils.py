import torch, numpy as np, os

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from purias_utils.util.logging import fixed_aspect_ratio
from purias_utils.util.plotting import lighten_color

from purias_utils.error_modelling_torus.non_parametric_error_model.variational_approx import NonParametricSwapErrorsVariationalModel
from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model import NonParametricSwapErrorsGenerativeModel


raise Exception('Dont use this file anymore please')



def sliced_mean_surface_2D(
    mean_surface, full_grid, swap_type, cued_feature_name, est_feature_name,
    axcuedslices_linear, axestimatedslices_linear, axcuedslices_exponentiated, axestimatedslices_exponentiated,
    min_separations, max_separations,
):
    """
    Two sets of slices for D=2, also plotting the exponential
    """
    cmap = plt.get_cmap('rainbow')
    cNorm  = colors.Normalize(vmin=-torch.pi, vmax=torch.pi)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    scalarMap.set_array([])

    assert swap_type == 'full'

    for g in range(min(mean_surface.shape)):
        col = scalarMap.to_rgba(full_grid[g])
        axcuedslices_linear.plot(full_grid, mean_surface[g,:], label = round(full_grid[g].item(), 3), c = col)
        axcuedslices_exponentiated.plot(full_grid, variational_model.normalisation_inner_numpy(mean_surface[g,:]), label = round(full_grid[g].item(), 3), c = col)
        axestimatedslices_linear.plot(full_grid, mean_surface[:,g], label = round(full_grid[g].item(), 3), c = col)
        axestimatedslices_exponentiated.plot(full_grid, variational_model.normalisation_inner_numpy(mean_surface[:,g]), label = round(full_grid[g].item(), 3), c = col)
    
    axcuedslices_linear.set_xlabel(f'Estimated feature ({est_feature_name})', fontsize = 20)
    axcuedslices_linear.set_title(f'$f$ slices, colours = cued feature ({cued_feature_name})', fontsize = 20)
    axestimatedslices_linear.set_xlabel(f'Cued feature ({cued_feature_name})', fontsize = 20)
    axestimatedslices_linear.set_title(f'$f$ slices, colours = estimated feature ({est_feature_name})', fontsize = 20)

    fixed_aspect_ratio(axcuedslices_linear, 1.0)
    fixed_aspect_ratio(axestimatedslices_linear, 1.0)

    axcuedslices_exponentiated.set_xlabel(f'Estimated feature ({est_feature_name})', fontsize = 20)
    axcuedslices_exponentiated.set_title(f'$e^f$ slices, colours = cued feature ({cued_feature_name})', fontsize = 20)
    axestimatedslices_exponentiated.set_xlabel(f'Cued feature ({cued_feature_name})', fontsize = 20)
    axestimatedslices_exponentiated.set_title(f'$e^f$ slices, colours = estimated feature ({est_feature_name})', fontsize = 20)

    fixed_aspect_ratio(axcuedslices_exponentiated, 1.0)
    fixed_aspect_ratio(axestimatedslices_exponentiated, 1.0)

    cax = axcuedslices_linear.inset_axes([1.04, 0.2, 0.05, 0.6], transform=axcuedslices_linear.transAxes)
    ticks = torch.linspace(-torch.pi+0.001, +torch.pi-0.001, 7)
    plt.colorbar(scalarMap, cax = cax, ticks=ticks, orientation='vertical')

    cax = axcuedslices_exponentiated.inset_axes([1.04, 0.2, 0.05, 0.6], transform=axcuedslices_exponentiated.transAxes)
    ticks = torch.linspace(-torch.pi+0.001, +torch.pi-0.001, 7)
    plt.colorbar(scalarMap, cax = cax, ticks=ticks, orientation='vertical')

    for ax_set in [[axcuedslices_linear, axestimatedslices_linear], [axcuedslices_exponentiated, axestimatedslices_exponentiated]]:
        for ax in ax_set:
            y_bot, y_top = ax.get_ylim()

            ax.plot([min_separations[0], min_separations[0]], [y_bot, y_top], color = 'black', linestyle = '--')
            ax.plot([-min_separations[0], -min_separations[0]], [y_bot, y_top], color = 'black', linestyle = '--')

            ax.plot([max_separations[0], max_separations[0]], [y_bot, y_top], color = 'black', linestyle = '--')
            ax.plot([-max_separations[0], -max_separations[0]], [y_bot, y_top], color = 'black', linestyle = '--')

            ax.set_ylim(y_bot, y_top)




def mean_and_variance_surface_1D(axes1D_linear, axes1D_exponentiated, x_label, mu, surface, upper_error_surface, lower_error_surface, sigma_chol, full_grid, min_separation, max_separation, inducing_points, inducing_points_means, pi_u_tilde, pi_1_tilde, normalisation_inner_function, f_evals = None, used_deltas = None, true_surface = None, all_deltas = None, num_samples = 10):
    """
    A single line for D=1. 
    Also plot the exponentiated, and include samples (linear and exponentiated).
    """
    
    if f_evals is not None: # [I, M, N+1]
        deltas = used_deltas[:,:,0].flatten().cpu().numpy()
        deltas_order = np.argsort(deltas)
        deltas = deltas[deltas_order]

        for f_eval in f_evals:      # [M, N]
            axes1D_linear.plot(deltas, f_eval.flatten()[deltas_order], alpha = 0.7, linewidth = 2)
            axes1D_exponentiated.plot(deltas, normalisation_inner_function(f_eval).flatten()[deltas_order], alpha = 0.7, linewidth = 2)

    full_grid = full_grid.numpy()
    surface = surface.flatten()
    lower_error_surface = lower_error_surface.flatten()
    upper_error_surface = upper_error_surface.flatten()

    lin_lines = axes1D_linear.plot(full_grid, surface, color = 'blue')
    axes1D_linear.fill_between(full_grid, lower_error_surface, upper_error_surface, color = lin_lines[0].get_color(), alpha = 0.2)
    sample_colour = lighten_color(lin_lines[0].get_color(), 1.6)
    for sample in f_samples:
        axes1D_linear.plot(full_grid, sample.numpy(), color = sample_colour, alpha = 0.3)
    axes1D_linear.plot(inducing_points, inducing_points_means.numpy(), color = 'black', linestyle = '--', marker = 'x', linewidth = 3, markersize = 12)
    axes1D_linear.plot([-torch.pi, torch.pi], [pi_u_tilde.item(), pi_u_tilde.item()], lin_lines[0].get_color(), linestyle= '-.', linewidth = 3)
    axes1D_linear.plot([-torch.pi, torch.pi], [pi_1_tilde.item(), pi_1_tilde.item()], lin_lines[0].get_color(), linewidth = 3)

    exp_lines = axes1D_exponentiated.plot(full_grid, normalisation_inner_function(surface).numpy(), color = 'red')
    axes1D_exponentiated.fill_between(full_grid, normalisation_inner_function(lower_error_surface).numpy(), normalisation_inner_function(upper_error_surface).numpy(), color = exp_lines[0].get_color(), alpha = 0.2)
    sample_colour = lighten_color(exp_lines[0].get_color(), 1.6)
    for sample in f_samples:
        axes1D_exponentiated.plot(full_grid, normalisation_inner_function(sample).numpy(), color = sample_colour, alpha = 0.3)
    axes1D_exponentiated.plot(inducing_points, normalisation_inner_function(inducing_points_means).numpy(), color = 'black', linestyle = '--', marker = 'x', linewidth = 3, markersize = 12)
    axes1D_exponentiated.plot([-torch.pi, torch.pi], [normalisation_inner_function(pi_u_tilde).item(), normalisation_inner_function(pi_u_tilde).item()], exp_lines[0].get_color(), linestyle= '-.', linewidth = 3)
    axes1D_exponentiated.plot([-torch.pi, torch.pi], [normalisation_inner_function(pi_1_tilde).item(), normalisation_inner_function(pi_1_tilde).item()], exp_lines[0].get_color(), linewidth = 3)

    if true_surface is not None:
        viz_x = all_deltas.squeeze(-1).cpu().numpy().flatten()
        viz_y = true_surface.flatten()
        viz_y[np.isneginf(viz_y)] = np.nan
        
        viz_x_order = np.argsort(viz_x)
        viz_x = viz_x[viz_x_order]
        viz_y = viz_y[viz_x_order]

        axes1D_linear.plot(viz_x, viz_y, color = lighten_color(lin_lines[0].get_color(), 2.0))
        axes1D_exponentiated.plot(viz_x, swap_function.normalisation_inner_numpy(viz_y), color = lighten_color(exp_lines[0].get_color(), 2.0))

    axes1D_linear.set_xlabel(x_label)
    axes1D_exponentiated.set_xlabel(x_label)
    
    axes1D_linear.set_xlabel(x_label)
    axes1D_linear.set_title("$p(f)$ with samples", fontsize = 25)
    axes1D_exponentiated.set_title("$e^{p(f)}$ with samples", fontsize = 25)

    for ax in [axes1D_linear, axes1D_exponentiated]:
        y_bot, y_top = ax.get_ylim()

        ax.plot([min_separation, min_separation], [y_bot, y_top], color = 'black', linestyle = '--')
        ax.plot([-min_separation, -min_separation], [y_bot, y_top], color = 'black', linestyle = '--')

        ax.plot([max_separation, max_separation], [y_bot, y_top], color = 'black', linestyle = '--')
        ax.plot([-max_separation, -max_separation], [y_bot, y_top], color = 'black', linestyle = '--')

        ax.set_ylim(y_bot, y_top)
        ax.set_xlim(-torch.pi, torch.pi)



def mean_and_variance_surface_2D(fig, ax3d_linear, ax3d_exponentiated, axheat_linear, axheat_exponentiated, inducing_points, grid_x, grid_y, surface, upper_error_surface, lower_error_surface, cued_feature_name, est_feature_name,):
    """
    A surface plot and imshow for D=2. 
    Also plot the exponentiated for both.
    """

    ax3d_linear.plot_surface(grid_x, grid_y, surface, color='red', label = 'Normalised function')
    ax3d_linear.plot_surface(grid_x, grid_y, upper_error_surface, color='red', alpha = 0.2)
    ax3d_linear.plot_surface(grid_x, grid_y, lower_error_surface, color='red', alpha = 0.2)
    ax3d_linear.set_title("$p(f)$", fontsize = 25)
    ax3d_linear.set_xlabel(f'Cued ({cued_feature_name})', fontsize = 20)
    ax3d_linear.set_ylabel(f'Estimated ({est_feature_name})', fontsize = 20)
    # if args.dataset_name == "misattribution_analysis":
    #     for N, dg in dataset_generator.data_generators.items():
    #         real_surface = dg.target_function(grid_points).reshape(grid_x.shape).detach().cpu().numpy()
    #         ax3d_linear.plot_surface(grid_x, grid_y, real_surface, color='blue', alpha = 0.3, label = 'N')
    #     # ax3d_linear.legend('$f^*$ for N = ...')

    ax3d_exponentiated.plot_surface(grid_x, grid_y, variational_model.normalisation_inner_numpy(surface), color='red', label = 'Normalised function')
    ax3d_exponentiated.plot_surface(grid_x, grid_y, variational_model.normalisation_inner_numpy(upper_error_surface), color='red', alpha = 0.2)
    ax3d_exponentiated.plot_surface(grid_x, grid_y, variational_model.normalisation_inner_numpy(lower_error_surface), color='red', alpha = 0.2)
    ax3d_exponentiated.set_title("$e^{p(f)}$", fontsize = 25)
    ax3d_exponentiated.set_xlabel(f'Cued ({cued_feature_name})', fontsize = 20)
    ax3d_exponentiated.set_ylabel(f'Estimated ({est_feature_name})', fontsize = 20)


    im = axheat_linear.imshow(surface.T, extent=[-torch.pi,+torch.pi,-torch.pi,+torch.pi], aspect = 'equal', cmap='inferno')
    axheat_linear.scatter(*inducing_points, color = 'red')
    divider = make_axes_locatable(axheat_linear)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axheat_linear.set_title("$E[f]$", fontsize = 25)
    axheat_linear.set_xlabel(f'Cued ({cued_feature_name})', fontsize = 20)
    axheat_linear.set_ylabel(f'Estimated ({est_feature_name})', fontsize = 20)

    im = axheat_exponentiated.imshow(surface.T, extent=[-torch.pi,+torch.pi,-torch.pi,+torch.pi], aspect = 'equal', cmap='inferno')
    axheat_exponentiated.scatter(*inducing_points, color = 'red')
    divider = make_axes_locatable(axheat_exponentiated)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    axheat_exponentiated.set_title("$e^{E[f]}$", fontsize = 25)
    axheat_exponentiated.set_xlabel(f'Cued ({cued_feature_name})', fontsize = 20)
    axheat_exponentiated.set_ylabel(f'Estimated ({est_feature_name})', fontsize = 20)




def plot_emission_distribution(emission_type, recent_delta_distributions, all_concentrations, generative_model, emissions_ax, device):

    raise Exception

    if emission_type == 'residual_deltas':

        fig_residual_deltas, axes_residual_deltas = plt.subplots(1, len(recent_delta_distributions), figsize = (5*len(recent_delta_distributions), 5))
        if len(recent_delta_distributions) == 1:
            axes_residual_deltas = [axes_residual_deltas]
        
        for si, (k, v) in enumerate(recent_delta_distributions.items()):
            
            alpha = 1 / (len(v) + 1)
            color = None
            label = k

            # Plot recent ones with alpha decreasing backwards:
            for locations, weights in v:
                theta_axis, pdf_estimate = generative_model.error_emissions.evaluate_emissions_pdf_on_circle(
                    k, locations = locations, weights = weights, device = device
                )
                lines = emissions_ax.plot(theta_axis, pdf_estimate, color = color, alpha = alpha, label = label)
                
                alpha += 1 / (len(v) + 1)
                color = lines[0].get_color()
                label = None
            
            # For the most recent one, also generate a new figure that has the histogram
            axes_residual_deltas[si].hist(locations.cpu().numpy(), 50, weights = weights.cpu().numpy(), density = True)
            axes_residual_deltas[si].plot(theta_axis, pdf_estimate, alpha = 1)

        return fig_residual_deltas

    else:
        for k, v in all_concentrations.items():
            emissions_ax.plot(v, label = f'N = {k}')

        emissions_ax.set_title(
            'Concentration' if emission_type == 'von_mises'
            else 'Alpha (first), Gamma (second)'
        )