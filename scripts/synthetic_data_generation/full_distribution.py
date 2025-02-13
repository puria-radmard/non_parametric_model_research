# Load up a model and its training data, then go over and generate data from an integral over the full q(f | \psi)

import numpy as np
import os, torch, argparse

from typing import Dict, Union

from matplotlib import pyplot as plt

from torch import Tensor as _T

# from non_parametric_model.scripts.main.setup import *
from purias_utils.util.arguments_yaml import ConfigNamepace
from purias_utils.error_modelling_torus.non_parametric_error_model.setup_utils import setup_model_whole, WorkingMemoryFullSwapModel, WorkingMemorySimpleSwapModel
from purias_utils.error_modelling_torus.data_utils.loading_utils import load_experimental_data

from purias_utils.maths.circular_statistics import kurtosis_from_angles, mean_resultant_length_from_angles
from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles


from non_parametric_model.scripts.synthetic_data_generation import utils



parser = argparse.ArgumentParser()
parser.add_argument('--resume_path', type = str)
parser.add_argument('--synthetic_data_code', type = str, required = True)
parser.add_argument('--num_synthetic_generation_repeats', type = int, required = True)
parser.add_argument('--allow_model_drop', action = 'store_true')
data_gen_args = parser.parse_args()

resume_path = data_gen_args.resume_path
args = ConfigNamepace.from_yaml_path(os.path.join(resume_path, "args.yaml"))
args.dict.pop('resume_path')

dataset_generator = load_experimental_data(args.dataset_name, args.train_indices_seed, args.train_indices_path, args.M_batch, args.M_test_per_set_size, args.num_models, args, 'cpu')
all_set_sizes = list(dataset_generator.data_generators.keys())

swap_model, D, delta_dimensions = setup_model_whole(
    **args.dict, all_set_sizes=all_set_sizes, trainable_kernel_delta=False, all_min_seps=None, num_variational_samples=args.monte_carlo_gp_samples, num_importance_sampling_samples=args.monte_carlo_gp_samples, resume_path=resume_path
)
swap_model: Union[WorkingMemoryFullSwapModel, WorkingMemorySimpleSwapModel]
swap_model.cuda()


data_destination = os.path.join(resume_path, f'synthetic_data_{data_gen_args.synthetic_data_code}.npy')
figure_destination = os.path.join(resume_path, f'synthetic_data_f_mean_evaluated_for_{data_gen_args.synthetic_data_code}.png')
config_yaml_destination = os.path.join(resume_path, f'synthetic_data_args_{data_gen_args.synthetic_data_code}.yaml')

assert resume_path is not None


for dest in [data_destination, figure_destination, config_yaml_destination]:
    if os.path.exists(dest):
        raise Exception('Cannot overwrite data! ' + dest)


dataset_sizes = {N: dg.all_deltas.shape[0] for N, dg in dataset_generator.data_generators.items()}
num_reps = data_gen_args.num_synthetic_generation_repeats


if args.num_models != 1:
    assert data_gen_args.allow_model_drop, "function_augmentation.py only supports Q=1, and num_synthetic_generation_repeats (K) will take the place of Q in the later stages! Pass --allow_model_drop if you want to allow this"
    swap_model.reduce_to_single_model()


# Adding extra dimension to allow direct compatibility with sweep_circular_statistics
all_mean_function_eval = {N: np.zeros([num_reps, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}
all_std_function_eval = {N: np.zeros([num_reps, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}
all_synthetic_prior_vectors = {N: np.zeros([num_reps, dataset_sizes[N], N + 1]).astype(float) for N in all_set_sizes}
all_synthetic_components = {N: np.zeros([num_reps, dataset_sizes[N]]) for N in all_set_sizes}
all_synthetic_estimate_errors = {N: np.zeros([num_reps, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}



with torch.no_grad():

    for rep_idx in range(num_reps):

        for i, (set_size, dg) in enumerate(dataset_generator.data_generators.items()):
            
            all_deltas: _T = dg.all_deltas.cuda()
            component_estimate_deltas = all_deltas[...,1].cuda()
            all_relevant_deltas = all_deltas[None,...,delta_dimensions].cuda() # Q = 1

            if args.swap_type == 'spike_and_slab':
                new_synth_data_batch = swap_model.generative_model.full_data_generation(set_size = set_size, vm_means = component_estimate_deltas, kwargs_for_generate_pi_vectors = {'batch_size': component_estimate_deltas.shape[0]})

            else:
                # # # inference_information = swap_model.inference(deltas=all_relevant_deltas)    # [Q (1), 4800]
                # # # f_samples = variational_model.reparameterised_sample(
                # # #     num_samples = args.monte_carlo_gp_samples, mu = inference_information.mu, 
                # # #     sigma_chol = inference_information.sigma_chol, M = all_relevant_deltas.shape[1], N = all_relevant_deltas.shape[2]
                # # # )                                                           # [1, I, batch, N]

                # # # new_mean_function_eval = variational_model.reinclude_model_evals(
                # # #     inference_information.mu.unsqueeze(1), num_displays = all_relevant_deltas.shape[1], 
                # # #     set_size = all_relevant_deltas.shape[2], num_mc_samples = 1
                # # # )

                # # # new_std_function_eval = variational_model.reinclude_model_evals(
                # # #     torch.stack([ss.diag().sqrt() for ss in inference_information.sigma], 0).unsqueeze(1), num_displays = all_relevant_deltas.shape[1], 
                # # #     set_size = all_relevant_deltas.shape[2], num_mc_samples = 1
                # # # )

                elbo_information = swap_model.minibatched_inference(deltas=all_relevant_deltas, max_variational_batch_size = 0, take_samples = True)
                new_mean_function_eval = elbo_information.mean_surface      # [1, M, N]
                new_std_function_eval = elbo_information.std_surface        # [1, M, N]
                new_f_samples = elbo_information.f_samples                  # [1, I, M, N]

                kwargs_for_generate_pi_vectors = dict(model_evaulations = new_f_samples, make_spike_and_slab = False)
                new_synth_data_batch: Dict[str, _T] = swap_model.generative_model.full_data_generation(set_size = set_size, vm_means = component_estimate_deltas, kwargs_for_generate_pi_vectors = kwargs_for_generate_pi_vectors)

            # [M, N]
            synthetic_errors: _T = rectify_angles(new_synth_data_batch['samples'].squeeze(0).unsqueeze(-1) - component_estimate_deltas)
            synthetic_errors_2: _T = rectify_angles(new_synth_data_batch['samples'].squeeze(0).unsqueeze(-1) + dg.all_target_zetas[:,0].cuda() - dg.all_target_zetas.squeeze(-1).cuda())
            assert torch.isclose(synthetic_errors, synthetic_errors_2).all()

            all_mean_function_eval[set_size][rep_idx] = all_mean_function_eval[set_size][rep_idx] * np.nan if args.swap_type == 'spike_and_slab' else new_mean_function_eval.cpu().numpy()
            all_std_function_eval[set_size][rep_idx] = all_std_function_eval[set_size][rep_idx] * np.nan if args.swap_type == 'spike_and_slab' else new_std_function_eval.cpu().numpy()
            all_synthetic_prior_vectors[set_size][rep_idx] = new_synth_data_batch['pi_vectors'].squeeze(0).cpu().numpy()
            all_synthetic_components[set_size][rep_idx] = new_synth_data_batch['betas'].squeeze(0).cpu().numpy()
            all_synthetic_estimate_errors[set_size][rep_idx] = synthetic_errors.cpu().numpy()


np.save(
    data_destination,
    {
        'generated_data': {
            "function_eval_mean": all_mean_function_eval,
            "function_eval_std": all_std_function_eval,
            "component_priors": all_synthetic_prior_vectors,
            "components": all_synthetic_components,
            "errors": all_synthetic_estimate_errors,
        }
    }
)


args.write_to_yaml(config_yaml_destination)
