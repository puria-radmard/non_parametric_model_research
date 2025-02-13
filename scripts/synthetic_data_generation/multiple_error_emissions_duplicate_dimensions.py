# Really weird ablation - swap the probe and cue axes when calculating the swap probs and see if that makes a difference
# Load up a model and its training data, then go over and generate data from an integral over the full q(f | \psi)
# This swap function posterior is the same across all the data, but we do differentiate based on 

import numpy as np
import os, torch, argparse

from typing import Dict
from pathlib import Path

from matplotlib import pyplot as plt

from purias_utils.error_modelling_torus.non_parametric_error_model.variational_approx import NonParametricSwapErrorsVariationalModel

from torch import Tensor as _T

# from non_parametric_model.scripts.main.setup import *
from purias_utils.util.arguments_yaml import ConfigNamepace
from purias_utils.error_modelling_torus.non_parametric_error_model.setup_utils import setup_model_whole, MultipleErrorEmissionsWorkingMemoryFullSwapModel
from purias_utils.error_modelling_torus.data_utils.loading_utils import load_experimental_data

from purias_utils.maths.circular_statistics import kurtosis_from_angles, mean_resultant_length_from_angles
from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles


from non_parametric_model.scripts.synthetic_data_generation import utils



parser = argparse.ArgumentParser()
parser.add_argument('--resume_path', type = str)
parser.add_argument('--num_synthetic_generation_repeats', type = int, required = True)
parser.add_argument('--num_underlying_data_duplications', type = int, required = False, default = 1)
parser.add_argument('--swap_function_multiplier', type = float, required = False, default = 1.0)        # --> didn't work in the end!
parser.add_argument('--hierarchical_config_yaml', type = str, help="", required = False, default = "/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/h_hierarchical_tests/hierarchical_config_gaussian.yaml")
parser.add_argument('--data_output_path', type = str, help = 'Since we are not using one model, we need an output math for the data! It will include signatures back to the models used')
parser.add_argument('--allow_model_drop', action = 'store_true')
data_gen_args = parser.parse_args()

resume_path = data_gen_args.resume_path
args = ConfigNamepace.from_yaml_path(os.path.join(resume_path, "args.yaml"))
extra_args = ConfigNamepace.from_yaml_path(data_gen_args.hierarchical_config_yaml)
args.dict.pop('resume_path')


dataset_generator = load_experimental_data(args.dataset_name, args.train_indices_seed, args.train_indices_path, args.M_batch, args.M_test_per_set_size, args.num_models, args, 'cpu')
all_set_sizes = list(dataset_generator.data_generators.keys())


synthetic_tag = 'different_error_emissions' + ('' if data_gen_args.num_underlying_data_duplications == 1 else f'_times{data_gen_args.num_underlying_data_duplications}')
if data_gen_args.swap_function_multiplier != 1.0:
    synthetic_tag += f'_mult{data_gen_args.swap_function_multiplier}'

dataset_generator.duplicate_underlying_data(data_gen_args.num_underlying_data_duplications)


separation_metadata_name = extra_args.separation_metadata_name
error_emissions_keys_across_set_sizes = [set(dg.all_metadata_inverted[separation_metadata_name].keys()) for N, dg in dataset_generator.data_generators.items()]
error_emissions_keys_across_set_sizes = list(set.union(*error_emissions_keys_across_set_sizes))

if args.drop_metadata_values is not None:
    for eek in args.drop_metadata_values:
        assert eek in error_emissions_keys_across_set_sizes, eek
        dataset_generator.drop_data_by_metadata_value(metadata_selection_key=separation_metadata_name, metadata_selection_value=eek)
        error_emissions_keys_across_set_sizes.remove(eek)

swap_model, D, delta_dimensions = setup_model_whole(
    **args.dict, error_emissions_keys=error_emissions_keys_across_set_sizes,
    all_set_sizes=all_set_sizes, trainable_kernel_delta=False, all_min_seps=None,
    num_variational_samples=args.monte_carlo_gp_samples,
    num_importance_sampling_samples=args.monte_carlo_gp_samples, resume_path=resume_path
)

##### 2025.02.13
assert len(delta_dimensions) == 1
other_delta_dimensions = [1 - delta_dimensions[0]]


kept_model_idx = 0

function_means_at_zero = {}
for set_size in dataset_generator.data_generators.keys():
    smiog = swap_model.inference_on_grid(set_size, 3, )
    assert smiog['one_dimensional_grid'][1] == 0.0
    function_means_at_zero[set_size] = torch.tensor(smiog['mean_surface'][[kept_model_idx],1]).cuda()
#####

swap_model: MultipleErrorEmissionsWorkingMemoryFullSwapModel
swap_model.cuda()


data_output_path = data_gen_args.data_output_path
Path(data_output_path).mkdir(parents=True, exist_ok=True)
data_destination = os.path.join(data_output_path, f'synthetic_data_different_error_emissions_swap_dimensions.npy')
config_yaml_destination = os.path.join(data_output_path, f'synthetic_data_args.yaml')
training_yaml_destination = os.path.join(data_output_path, f'synthetic_data_training_args.yaml')


assert resume_path is not None


for dest in [data_destination, config_yaml_destination]:
    if os.path.exists(dest):
        print('Cannot overwrite data! ' + dest)
        # raise Exception('Cannot overwrite data! ' + dest)


dataset_sizes = {N: dg.all_deltas.shape[0] for N, dg in dataset_generator.data_generators.items()}
num_reps = data_gen_args.num_synthetic_generation_repeats


if args.num_models != 1:
    assert data_gen_args.allow_model_drop, "function_augmentation.py only supports Q=1, and num_synthetic_generation_repeats (K) will take the place of Q in the later stages! Pass --allow_model_drop if you want to allow this"
    swap_model.reduce_to_single_model(kept_model_idx)


# Adding extra dimension to allow direct compatibility with sweep_circular_statistics
all_mean_function_eval = {N: np.zeros([num_reps, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}
all_std_function_eval = {N: np.zeros([num_reps, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}
all_synthetic_prior_vectors = {N: np.zeros([num_reps, dataset_sizes[N], N + 1]).astype(float) for N in all_set_sizes}
all_synthetic_components = {N: np.zeros([num_reps, dataset_sizes[N]]) for N in all_set_sizes}
all_synthetic_estimate_errors = {N: np.zeros([num_reps, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}


assert args.swap_type != 'spike_and_slab'


with torch.no_grad():

    for rep_idx in range(num_reps):

        for i, (set_size, dg) in enumerate(dataset_generator.data_generators.items()):

            for error_emissions_key in error_emissions_keys_across_set_sizes:
            
                all_smk_indices = dg.all_metadata_inverted[separation_metadata_name][error_emissions_key]
                all_error_emissions_errors = dg.all_errors[:,all_smk_indices,:].to('cuda')
                error_emissions_component_estimate_deltas = dg.all_deltas[all_smk_indices,:,1].cuda()
                
                ##### 2025.02.13 ###############################################################################################################################################
                all_error_emissions_relevant_deltas_original = dg.all_deltas[all_smk_indices][...,delta_dimensions].unsqueeze(0).repeat(swap_model.num_models, 1, 1, 1).to('cuda')    # [Q, M_s, N, D]
                elbo_information_original = swap_model.minibatched_inference(deltas=all_error_emissions_relevant_deltas_original, max_variational_batch_size = 0, take_samples = True)
                
                all_error_emissions_relevant_deltas_other = dg.all_deltas[all_smk_indices][...,other_delta_dimensions].unsqueeze(0).repeat(swap_model.num_models, 1, 1, 1).to('cuda')    # [Q, M_s, N, D]
                elbo_information_other = swap_model.minibatched_inference(deltas=all_error_emissions_relevant_deltas_other, max_variational_batch_size = 0, take_samples = True)


                mean_at_zero = function_means_at_zero[set_size][:,None,None]

                # elbo_information_original.mean_surface[...,1:].min()
                # dim_duped_mean_surface[...,1:].min()

                dim_duped_mean_surface = data_gen_args.swap_function_multiplier * (
                    elbo_information_original.mean_surface + elbo_information_other.mean_surface
                ) / mean_at_zero
                new_std_function_eval = elbo_information_original.std_surface

                # Get the samples of f evaluated at the deltas
                all_f_samples = [
                    swap_model.get_variational_model(set_size).reparameterised_sample(
                        num_samples = swap_model.num_variational_samples,
                        mu = data_gen_args.swap_function_multiplier * (mu + mu_other) / mean_at_zero, 
                        sigma_chol = sigma_chol,
                        M = M, N = set_size
                    )
                    for mu, mu_other, sigma_chol, M in zip(
                        elbo_information_original.mus, elbo_information_other.mus, elbo_information_original.sigma_chols, elbo_information_original.M_minis
                    )
                ]   # Each of shape [Q, K, ~batchsize, N]                

                dim_duped_f_samples = torch.concat(all_f_samples, 2)  # [Q, I, M, N]

                kwargs_for_generate_pi_vectors = dict(model_evaulations = dim_duped_f_samples, make_spike_and_slab = False)
                ################################################################################################################################################################

                new_synth_data_batch: Dict[str, _T] = swap_model.generative_model.full_data_generation(error_emissions_key = error_emissions_key, set_size = set_size, vm_means = error_emissions_component_estimate_deltas, kwargs_for_generate_pi_vectors = kwargs_for_generate_pi_vectors)

                # [M, N]
                synthetic_errors: _T = rectify_angles(new_synth_data_batch['samples'].squeeze(0).unsqueeze(-1) - error_emissions_component_estimate_deltas)
                synthetic_errors_2: _T = rectify_angles(new_synth_data_batch['samples'].squeeze(0).unsqueeze(-1) + dg.all_target_zetas[all_smk_indices,0].cuda() - dg.all_target_zetas[all_smk_indices].squeeze(-1).cuda())
                if not torch.isclose(synthetic_errors, synthetic_errors_2).all():
                    print((synthetic_errors - synthetic_errors_2).abs().max())

                all_mean_function_eval[set_size][rep_idx,all_smk_indices] = all_mean_function_eval[set_size][rep_idx] * np.nan if args.swap_type == 'spike_and_slab' else dim_duped_mean_surface.cpu().numpy()
                all_std_function_eval[set_size][rep_idx,all_smk_indices] = all_std_function_eval[set_size][rep_idx] * np.nan if args.swap_type == 'spike_and_slab' else new_std_function_eval.cpu().numpy()
                all_synthetic_prior_vectors[set_size][rep_idx,all_smk_indices] = new_synth_data_batch['pi_vectors'].squeeze(0).cpu().numpy()
                all_synthetic_components[set_size][rep_idx,all_smk_indices] = new_synth_data_batch['betas'].squeeze(0).cpu().numpy()
                all_synthetic_estimate_errors[set_size][rep_idx,all_smk_indices] = synthetic_errors.cpu().numpy()


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

training_yaml_info = {
    'dataset_name': 'cross_model_fit',
    'underlying_dataset': args.dataset_name,
    'synthetic_data_root': str(Path(data_output_path).parent.absolute()),
    'synthetic_data_code': 'different_error_emissions_swap_dimensions',
    'num_underlying_data_duplications': data_gen_args.num_underlying_data_duplications,
    'generation_path': resume_path
}

ConfigNamepace(training_yaml_info).write_to_yaml(training_yaml_destination)


