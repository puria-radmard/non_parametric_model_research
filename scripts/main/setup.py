"""
If looking for old VM function model recovery, see archive/non_parametric_model/training.py
"""

from non_parametric_model.scripts.main.args import *
from purias_utils.error_modelling_torus.non_parametric_error_model.setup_utils import setup_model_whole

import torch
import numpy as np
from collections import deque

from purias_utils.util.logging import LoopTimer, EarlyStopper

from purias_utils.error_modelling_torus.data_utils.loading_utils import load_experimental_data, load_synthetic_data, dataset_choices



torch.autograd.set_detect_anomaly(True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


########## load data and make task ##########
if args.dataset_name in dataset_choices:
    dataset_generator = load_experimental_data(args.dataset_name, args.train_indices_seed, args.train_indices_path, M_batch, M_test_per_set_size, num_models, args, device = 'cpu')
    true_mean_surfaces_dict = true_std_surfaces_dict = {N: None for N in dataset_generator.data_generators.keys()}
    track_fmse: bool = False

elif args.dataset_name == 'cross_model_fit':
    assert args.train_indices_path == None, "model recovery no longer takes train_indices_path, please use synthetic_data_path and specify a synthetic_data_code and a train_indices_seed"
    assert args.train_indices_seed != None, "train_indices_seed is required for model recovery"
    assert args.train_examples_discarded_per_set_size == None, "args.train_examples_discarded_per_set_size needs reimplementing for new loading process!"
    dataset_generator = load_experimental_data(args.underlying_dataset, args.train_indices_seed, None, M_batch, M_test_per_set_size, num_models, args, device = 'cpu')
    dataset_generator, true_mean_surfaces_dict, true_std_surfaces_dict = load_synthetic_data(dataset_generator, args.synthetic_data_path, args.synthetic_data_code)
    track_fmse: bool = (args.swap_type != 'spike_and_slab')

    if args.discard_last_n_training_examples > 0:
        print([dg.discard_last_n_training_examples(int(args.discard_last_n_training_examples)) for dg in dataset_generator.data_generators.values()])


min_separations = {N: [dg.all_deltas[:,1:,d].abs().min().item() if N > 1 else 0.0 for d in range(dataset_generator.D)] for N, dg in dataset_generator.data_generators.items()}
max_separations = {N: [dg.all_deltas[:,1:,d].abs().max().item() if N > 1 else 0.0 for d in range(dataset_generator.D)] for N, dg in dataset_generator.data_generators.items()}
#############################################



################ init models ################
all_set_sizes = list(dataset_generator.data_generators.keys())

if args.init_min_seps:
    min_seps = torch.tensor([min_separations[N] for N in all_set_sizes])  # Avoid numerical issues possibly...
else:
    min_seps = None

swap_model, D, delta_dimensions = setup_model_whole(
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
    normalisation_inner= args.normalisation_inner,
    num_variational_samples = args.monte_carlo_gp_samples,
    num_importance_sampling_samples = args.monte_carlo_gp_samples,
    resume_path = args.resume_path,
    device=device,
)


opt = torch.optim.Adam(swap_model.parameters(), lr = args.lr)
#############################################



########### initialise logging #############
num_set_sizes = len(all_set_sizes)
biggest_M = max([dataset_generator.data_generators[N].M_train for N in all_set_sizes])
steps_per_epoch = int(np.ceil(biggest_M / M_batch)) if M_batch > 0 else 1
T = args.num_training_examples * num_set_sizes * steps_per_epoch

timer = LoopTimer(T)
early_stopper = EarlyStopper(T)

# One saved at each timestep for the relevant set size
training_step_per_set_size = {N: [] for N in all_set_sizes}
all_average_llh_losses_per_set_size = np.zeros([T, num_models, num_set_sizes])
all_average_elbos_per_set_size = np.zeros([T, num_models, num_set_sizes])
all_kl_losses_per_set_size = np.zeros([T, num_models, num_set_sizes])
all_dist_losses_per_set_size = np.zeros([T, num_models, num_set_sizes])

early_stopper.watch(all_average_elbos_per_set_size, 'performance')

# One saved every so often for all set sizes - derived from recent_total_XXX_log_likelihoods, defined below
test_save_steps = []
all_average_train_set_naive_log_likelihoods = np.zeros([(T // testing_frequency) + 1, num_models, num_set_sizes])
all_average_train_set_importance_sampled_log_likelihoods = np.zeros([(T // testing_frequency) + 1, num_models, num_set_sizes])
all_average_test_set_naive_log_likelihoods = np.zeros([(T // testing_frequency) + 1, num_models, num_set_sizes])
all_average_test_set_importance_sampled_log_likelihoods = np.zeros([(T // testing_frequency) + 1, num_models, num_set_sizes])
if track_fmse:
    all_scaled_rmse_from_true_confusion_function = np.zeros([(T // testing_frequency) + 1, num_models, num_set_sizes])

# As above, but for logging to json
recent_component_posteriors = {}
recent_component_priors = {}
recent_particle_non_uniform_weights = {}
recent_particle_uniform_weights = {}
recent_particle_mean_first_resultant_vector_lengths = {}
recent_particle_circular_kurtosis = {}
# As above, but for logging to json for each item
recent_naive_log_likelihoods = {}
recent_importance_sampled_log_likelihoods = {}

# One saved at each timestep for all set sizes
if swap_type == 'spike_and_slab':
    all_p_corrects = np.zeros([T, num_models, num_set_sizes])
    all_p_swaps = np.zeros([T, num_models, num_set_sizes])
    all_p_unifs = np.zeros([T, num_models, num_set_sizes])
    early_stopper.watch(all_p_corrects, 'parameter')
    early_stopper.watch(all_p_swaps, 'parameter')
    early_stopper.watch(all_p_unifs, 'parameter')
else:
    all_inverse_ells = np.zeros([T, num_models, num_set_sizes, D])
    all_scalers = np.zeros([T, num_models, num_set_sizes])
    all_pi_u_tildes = np.zeros([T, num_models, num_set_sizes]) * np.nan
    all_pi_1_tildes = np.zeros([T, num_models, num_set_sizes]) * np.nan
    early_stopper.watch(all_inverse_ells, 'parameter')
    early_stopper.watch(all_scalers, 'parameter')
    early_stopper.watch(all_pi_u_tildes, 'parameter')
    early_stopper.watch(all_pi_1_tildes, 'parameter')


scalar_plot_x_axis = np.arange(1, T+1)[None].repeat(num_models, 0)

num_emissions_parameters = swap_model.generative_model.error_emissions.emission_parameter(all_set_sizes[0]).shape[-1]
all_emissions_parameters = np.zeros([T, num_models, num_set_sizes, num_emissions_parameters])
early_stopper.watch(all_emissions_parameters, 'parameter')

if args.shared_swap_function:
    recent_delta_distributions = {0: deque([], maxlen = 10)}
else:
    recent_delta_distributions = {N: deque([], maxlen = 10) for N in all_set_sizes}



#############################################


########### figure out natural beta factor for ELBO #############
if beta == 'nat':

    betas = {}

    if args.M_batch > 0:

        if args.shared_swap_function:
            total_M_train = sum(dg.M_train for dg in dataset_generator.data_generators.values())
            betas = {N: dg.M_batch / total_M_train for N, dg in dataset_generator.data_generators.items()}
        else:    # Seperated by set size, so each ELBO responsible for only that set size's examples
            betas = {N: dg.M_batch / dg.M_train for N, dg in dataset_generator.data_generators.items()}
            
    else:

        if args.shared_swap_function:
            total_M_train = sum(dg.M_train for dg in dataset_generator.data_generators.values())
            betas = {N: dg.M_train / total_M_train for N, dg in dataset_generator.data_generators.items()}
        else:
            betas = {N: 1.0 for N, dg in dataset_generator.data_generators.items()}
#################################################################

swap_model.to(device)

