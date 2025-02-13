from purias_utils.error_modelling_torus.data_utils.loading_utils import load_experimental_data, dump_training_indices_to_path
from purias_utils.util.plotting import legend_without_repeats

from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model import NonParametricSwapErrorsGenerativeModel
from purias_utils.error_modelling_torus.non_parametric_error_model.main import WorkingMemorySimpleSwapModel

from pathlib import Path
from purias_utils.util.logging import configure_logging_paths

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from itertools import chain
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
from torch import Tensor as _T

device = 'cuda'

parser = argparse.ArgumentParser()

# Data args
parser.add_argument('--dataset_name', type = str, default='bays2009', required=False)
parser.add_argument('--train_indices_seed', type = int, required=False, default = 42)
parser.add_argument('--M_test_per_set_size', type = float, required=True)
parser.add_argument('--num_models', type = int, required=False, default = 32)
parser.add_argument('--subjects', type = int, nargs = '+', required=False, default = None)
parser.add_argument('--stimulus_exposure_id', type = int, nargs = '+', required=False, default = None)

# Only one model arg!
parser.add_argument('--emission_type', type = str, required=True)

# logging_args
parser.add_argument('--logging_base', type = str, required=True)
parser.add_argument('--run_name', type = str, required=False, default = 'run')



args = parser.parse_args()


### Setup

# Data
dataset_generator = load_experimental_data(args.dataset_name, args.train_indices_seed, None, 0, args.M_test_per_set_size, args.num_models, args, device = 'cuda')
assert 1 in dataset_generator.data_generators.keys(), f"Cannot do set_size_1 with dataset {args.dataset_name} - no set size 1 data"

# Model
remove_uniform = (args.emission_type != 'von_mises')
generative_model = NonParametricSwapErrorsGenerativeModel.from_typical_args(
    num_models = args.num_models,
    swap_type = 'spike_and_slab',
    kernel_type = None,
    emission_type = args.emission_type,
    fix_non_swap = None,
    remove_uniform = remove_uniform,
    include_pi_u_tilde = not remove_uniform,
    include_pi_1_tilde = False,
    normalisation_inner = 'exp',
    shared_swap_function = True,
    shared_emission_distribution = True,
    all_set_sizes = [1],
    trainable_kernel_delta = False,
    num_features = 0,
    device = device,
)
swap_model = WorkingMemorySimpleSwapModel(generative_model).to(device)
num_emissions_parameters = swap_model.generative_model.error_emissions.emission_parameter(1).shape[-1]

# Training and plotting
opt = Adam(params = generative_model.parameters(), lr = 0.005)
num_batches = int(3000) if args.emission_type == 'wrapped_stable' else int(1e4)
training_llhs, testing_steps, testing_llhs = np.zeros([num_batches, args.num_models]), [], np.zeros([num_batches, args.num_models])
all_p_corrects, all_p_unifs = np.zeros([num_batches, args.num_models]), np.zeros([num_batches, args.num_models])
all_emissions_parameters = np.zeros([num_batches, args.num_models, num_emissions_parameters])
scalar_plot_x_axis = np.arange(1, num_batches+1)[None].repeat(args.num_models, 0)


# Logging
Path(args.logging_base).mkdir(parents = True, exist_ok = True)
logging_directory = os.path.join(args.logging_base, args.run_name)
[training_print_path, testing_print_path], logging_directory, _ = configure_logging_paths(logging_directory, log_suffixes=["train", "full"], index_new=True)
dump_training_indices_to_path(dataset_generator, logging_directory)
parameter_save_path = os.path.join(logging_directory, '{model}.{ext}')
with open(os.path.join(logging_directory, "args.json"), 'w') as jf:
    json.dump(vars(args), jf)
    
param_colors = ['blue', 'red']

### Training
t = -1

for batch_N, batch_M, deltas_batch, errors_batch, _ in tqdm(dataset_generator.iterate_train_batches(dimensions = ..., shuffle = True, total = num_batches, N=1), total = num_batches):

    t += 1

    assert (deltas_batch == 0.0).all() and (batch_N == 1)

    # Optimise
    opt.zero_grad()
    training_info = swap_model.get_elbo_terms_easier(errors_batch, batch_M, 1, 'error')
    train_llhs: _T = training_info['total_log_likelihood']
    (-train_llhs.sum()).backward()
    training_llhs[t,:] = train_llhs.detach().cpu().numpy() / batch_M
    opt.step()

    sands_pi_vector = swap_model.generative_model.swap_function.generate_pi_vectors(1, 1)['pis'].detach().cpu()
    all_p_unifs[t,:] = sands_pi_vector[:,0,0]
    all_p_corrects[t,:] = sands_pi_vector[:,0,1]
    all_emissions_parameters[t,:,:] = swap_model.generative_model.error_emissions.emission_parameter(1).detach().cpu()

    # Test
    if t % 200 == 0:
        testing_steps.append(t)

        with torch.no_grad():
            test_batch_M = 0
            for test_batch_N, test_batch_M, test_deltas_batch, test_errors_batch, batch_indices in dataset_generator.all_test_batches(dimensions = ..., N = 1):
                assert (test_deltas_batch == 0.0).all() and (test_batch_N == 1)
                testing_info = swap_model.get_elbo_terms_easier(test_errors_batch, test_batch_M, 1, 'error')
                test_llhs: _T = testing_info['total_log_likelihood']
                testing_llhs[t,:] = test_llhs.detach().cpu().numpy() / test_batch_M
        
        torch.save(swap_model.state_dict(), parameter_save_path.format(model = 'swap_model', ext = 'mdl'))

        recent_llhs_dict = {
            "average_training_llh": training_llhs[t,:],
            "average_testing_llh": testing_llhs[t,:],
        }
        np.save(os.path.join(logging_directory, "recent_llhs_dict.npy"), recent_llhs_dict)

        assert batch_M + test_batch_M == dataset_generator.data_generators[1].all_deltas.shape[0]

        # Plot
        fig, axes = plt.subplots(2, 2, figsize = (16, 16))
        axes = axes.flatten()
        axes[0].set_title('Uniform probability (ignore if not von Mises + uniform)')
        axes[1].set_title('Emission parameters')
        axes[2].set_title('Training and testing log-likelihoods (per item)')
        axes[3].set_title('Overall PDF')

        axes[0].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,:t+1], all_p_corrects[:t+1,:].T))), label = f'p(correct)', c = 'blue', alpha = 0.4)
        axes[0].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,:t+1], all_p_unifs[:t+1,:].T))), label = f'p(uniform)', c = 'red', alpha = 0.4)
        legend_without_repeats(axes[0])

        for iii in range(num_emissions_parameters):
            param_color = axes[1].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,:t+1], all_emissions_parameters[:t+1,:,iii].T))), label = f'Param {iii}', c = param_colors[iii], alpha = 0.4)
        legend_without_repeats(axes[1])

        axes[2].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,:t+1], training_llhs[:t+1,:].T))), label = 'Train', c = 'blue', alpha = 0.4)
        axes[2].plot(*list(chain.from_iterable(zip(scalar_plot_x_axis[:,testing_steps], testing_llhs[testing_steps,:].T))), label = 'Test', c = 'red', alpha = 0.4)
        legend_without_repeats(axes[2])

        example_deltas_batch = dataset_generator.data_generators[1].all_deltas[[0]].unsqueeze(0).repeat(swap_model.generative_model.num_models, 1, 1, 1)
        example_target_zetas_batch = dataset_generator.data_generators[1].all_target_zetas[[0]]
        theta_axis, pdfs = swap_model.visualise_pdf_for_example(example_deltas_batch, example_target_zetas_batch, 360)
        for pdf in pdfs:
            axes[3].plot(theta_axis.cpu().numpy(), pdf.cpu().numpy())
        legend_without_repeats(axes[3])

        fig.savefig(os.path.join(logging_directory, "log.png"))
        plt.close('all')
