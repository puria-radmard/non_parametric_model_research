from non_parametric_model.scripts.main.args import *

from purias_utils.error_modelling_torus.non_parametric_error_model.setup_utils import setup_model_whole_hierarchical

import torch
import numpy as np
from collections import deque

from purias_utils.util.logging import LoopTimer, EarlyStopper

from purias_utils.error_modelling_torus.data_utils.loading_utils import load_experimental_data, load_synthetic_data_hierarchical, dataset_choices




separation_metadata_name = args.separation_metadata_name
assert separation_metadata_name is not None, "Require some separation_metadata_name to run hierarchical script! Typically this is 'subject_idx'"
assert swap_type != 'spike_and_slab'
# assert num_models == 1, "Viz doesn't work with num_models > 1!"


torch.autograd.set_detect_anomaly(True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


########## load data and make task ##########
assert M_test_per_set_size == 0, "Need to be more sure about test sets for hierarchical model!"

if args.dataset_name in dataset_choices:
    dataset_generator = load_experimental_data(args.dataset_name, args.train_indices_seed, args.train_indices_path, M_batch, M_test_per_set_size, num_models, args, device = 'cpu')
    true_primary_surfaces = true_primary_surfaces_std = true_submodel_surfaces = true_submodel_surfaces_std = true_total_surfaces = true_total_surfaces_std =\
         {N: None for N in dataset_generator.data_generators.keys()}

elif args.dataset_name == 'cross_model_fit':
    assert args.train_indices_path == None, "model recovery no longer takes train_indices_path, please use synthetic_data_path and specify a synthetic_data_code and a train_indices_seed"
    assert args.train_indices_seed != None, "train_indices_seed is required for model recovery"
    assert args.train_examples_discarded_per_set_size == None, "args.train_examples_discarded_per_set_size needs reimplementing for new loading process!"
    dataset_generator = load_experimental_data(args.underlying_dataset, args.train_indices_seed, None, M_batch, M_test_per_set_size, num_models, args, device = 'cpu')
    dataset_generator, true_primary_surfaces, true_primary_surfaces_std, true_submodel_surfaces, true_submodel_surfaces_std, true_total_surfaces, true_total_surfaces_std =\
        load_synthetic_data_hierarchical(dataset_generator, args.synthetic_data_path, args.synthetic_data_code)
    track_fmse: bool = False    # Forget it!

    if args.discard_last_n_training_examples > 0:
        raise NotImplementedError('Need to think more about discard_last_n_training_examples for hierarchical model')



min_separations = {N: [dg.all_deltas[:,1:,d].abs().min().item() if N > 1 else 0.0 for d in range(dataset_generator.D)] for N, dg in dataset_generator.data_generators.items()}
max_separations = {N: [dg.all_deltas[:,1:,d].abs().max().item() if N > 1 else 0.0 for d in range(dataset_generator.D)] for N, dg in dataset_generator.data_generators.items()}

# If these aren't the same for all setsizes, some of the parameters might not get trained! 
# This is a matter of self.all_metadata design then...
submodel_keys_across_set_sizes = [set(dg.all_metadata_inverted[separation_metadata_name].keys()) for N, dg in dataset_generator.data_generators.items()]
submodel_keys_across_set_sizes = list(set.union(*submodel_keys_across_set_sizes))
num_submodels = len(submodel_keys_across_set_sizes)
#############################################


################ init models ################
all_set_sizes = list(dataset_generator.data_generators.keys())

if args.init_min_seps:
    min_seps = torch.tensor([min_separations[N] for N in all_set_sizes])  # Avoid numerical issues possibly...
else:
    min_seps = None


TEST_PRIMARY_MODEL_RESUME_PATH = None# '/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_summaries_7_10_24/mcmaster2022_e1_oricue_cue_AR2/cue_dim_wrapped_stable_0'

swap_model, D, delta_dimensions = setup_model_whole_hierarchical(
    submodel_keys = submodel_keys_across_set_sizes.copy(),
    num_models = num_models,
    swap_type = swap_type,
    kernel_type = args.kernel_type,
    emission_type = emission_type,
    all_set_sizes = all_set_sizes,
    remove_uniform = remove_uniform,
    include_pi_u_tilde = include_pi_u_tilde,
    trainable_kernel_delta = trainable_kernel_delta,
    R_per_dim = args.R_per_dim,
    fix_non_swap = fix_non_swap,
    include_pi_1_tilde = include_pi_1_tilde,
    fix_inducing_point_locations = fix_inducing_point_locations,
    symmetricality_constraint = args.symmetricality_constraint,
    shared_swap_function = args.shared_swap_function,
    shared_emission_distribution = args.shared_emission_distribution,
    all_min_seps = min_seps,
    inducing_point_variational_parameterisation_type=args.inducing_point_variational_parameterisation_type,
    inducing_point_variational_submodel_parameterisation_type=args.inducing_point_variational_submodel_parameterisation_type,
    normalisation_inner= args.normalisation_inner,
    num_variational_samples = args.monte_carlo_gp_samples,
    num_importance_sampling_samples = args.monte_carlo_gp_samples,
    resume_path = args.resume_path,
    primary_model_resume_path = TEST_PRIMARY_MODEL_RESUME_PATH,
    device=device,
)


opt = torch.optim.Adam(swap_model.parameters(), lr = args.lr)
#############################################



if args.drop_metadata_values is not None:
    for smk in args.drop_metadata_values:
        assert smk in submodel_keys_across_set_sizes, smk
        dataset_generator.drop_data_by_metadata_value(metadata_selection_key=separation_metadata_name, metadata_selection_value=smk)
        swap_model.drop_submodel(smk)
        submodel_keys_across_set_sizes.remove(smk)
        num_submodels -= 1


########### initialise logging #############
num_set_sizes = len(all_set_sizes)
biggest_M = max([dataset_generator.data_generators[N].M_train for N in all_set_sizes])
steps_per_epoch = int(np.ceil(biggest_M / M_batch)) if M_batch > 0 else 1
T = args.num_training_examples * num_set_sizes * steps_per_epoch * len(submodel_keys_across_set_sizes)

timer = LoopTimer(T)
print('No early stopper!')

training_step_per_set_size = {N: [] for N in all_set_sizes}
training_step_per_submodel_key = {N: {smk: [] for smk in submodel_keys_across_set_sizes} for N in all_set_sizes}
all_average_llh_losses_per_set_size = np.zeros([T, num_set_sizes, num_submodels])
all_average_elbos_per_set_size = np.zeros([T, num_set_sizes, num_submodels])
all_primary_kl_losses_per_set_size = np.zeros([T, num_set_sizes])
all_submodel_kl_losses_per_set_size = np.zeros([T, num_set_sizes, num_submodels])
all_dist_losses_per_set_size = np.zeros([T, num_set_sizes, num_submodels])

test_save_steps = []
all_average_train_set_naive_log_likelihoods = np.zeros([(T // testing_frequency) + 1, num_set_sizes])

recent_naive_log_likelihoods = {N: {} for N in all_set_sizes}
recent_component_priors = {N: {} for N in all_set_sizes}

all_primary_inverse_ells = np.zeros([T, num_set_sizes, D])
all_primary_scalers = np.zeros([T, num_set_sizes])
all_submodel_inverse_ells = np.zeros([T, num_set_sizes, num_submodels, D])
all_submodel_scalers = np.zeros([T, num_set_sizes, num_submodels])
all_pi_u_tildes = np.zeros([T, num_set_sizes, num_submodels]) * np.nan
all_pi_1_tildes = np.zeros([T, num_set_sizes, num_submodels]) * np.nan


scalar_plot_x_axis = np.arange(1, T+1)

num_emissions_parameters = swap_model.get_generative_submodel(submodel_keys_across_set_sizes[0]).error_emissions.emission_parameter(all_set_sizes[0]).shape[-1]
all_emissions_parameters = np.zeros([T, num_set_sizes, num_submodels, num_emissions_parameters])


########### dont need to implement all of this yet! #############
assert beta == 'nat', "Cannot change beta for hierarchical model yet!"
assert not args.shared_swap_function and not args.shared_emission_distribution, "Cannot do shared generative model for hierarchical model right now..."
#################################################################


swap_model.to(device)
