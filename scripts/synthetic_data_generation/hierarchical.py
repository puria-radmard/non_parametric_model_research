# For a hierarchical dataset, load up a master model and a set of submodels
    # args is a bit different for this one - see below
# NOTE: for now we are ignoring inducing locations for the submodels - just fix them to be where the primary model has them
# Then go over and generate data from an integral over the full <q(f | f_s, \psi_s)>_{q(f | \tilde\psi)}


import numpy as np
from tqdm import tqdm
import os, torch, argparse

from typing import Dict
from pathlib import Path

from torch import Tensor as _T

# from non_parametric_model.scripts.main.setup import *
from purias_utils.util.arguments_yaml import ConfigNamepace
from purias_utils.error_modelling_torus.non_parametric_error_model.setup_utils import setup_model_whole_hierarchical
from purias_utils.error_modelling_torus.data_utils.loading_utils import load_experimental_data
from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles



parser = argparse.ArgumentParser()
parser.add_argument('--primary_model_resume_path', type = str, help="Used for args also, in combination with hierarchical_config_yaml")
parser.add_argument('--hierarchical_config_yaml', type = str, help="", required = False, default = "/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/h_hierarchical_tests/hierarchical_config_gaussian.yaml")
parser.add_argument('--submodel_resume_path_template', type = str, help = 'e.g. if key is subject_idx, pass: ".../base_path/run_name_{subject_idx}", which will get formatted using arg below. Have to be careful that these match args given above!')
parser.add_argument('--data_output_path', type = str, help = 'Since we are not using one model, we need an output math for the data! It will include signatures back to the models used')
parser.add_argument('--num_synthetic_generation_repeats', type = int, required = True)
parser.add_argument('--allow_model_drop', action = 'store_true')
data_gen_args = parser.parse_args()


primary_model_resume_path = data_gen_args.primary_model_resume_path
args = ConfigNamepace.from_yaml_path(os.path.join(primary_model_resume_path, "args.yaml"))
extra_args = ConfigNamepace.from_yaml_path(data_gen_args.hierarchical_config_yaml)
args.update(extra_args)
args.dict.pop('resume_path')

dataset_generator = load_experimental_data(args.dataset_name, args.train_indices_seed, args.train_indices_path, args.M_batch, args.M_test_per_set_size, args.num_models, args, 'cpu')
all_set_sizes = list(dataset_generator.data_generators.keys())

separation_metadata_name = extra_args.separation_metadata_name
submodel_keys_across_set_sizes = [set(dg.all_metadata_inverted[separation_metadata_name].keys()) for N, dg in dataset_generator.data_generators.items()]
submodel_keys_across_set_sizes = list(set.union(*submodel_keys_across_set_sizes))


swap_model, D, delta_dimensions = setup_model_whole_hierarchical(
    **args.dict, all_set_sizes=all_set_sizes, trainable_kernel_delta=False, all_min_seps=None,
    submodel_keys = submodel_keys_across_set_sizes, resume_path = None,
    num_variational_samples=args.monte_carlo_gp_samples, num_importance_sampling_samples=args.monte_carlo_gp_samples,
    primary_model_resume_path = None
)

if args.num_models != 1:
    assert data_gen_args.allow_model_drop, "function_augmentation.py only supports Q=1, and num_synthetic_generation_repeats (K) will take the place of Q in the later stages! Pass --allow_model_drop if you want to allow this"
    swap_model.reduce_to_single_model()



dataset_sizes = {N: dg.all_deltas.shape[0] for N, dg in dataset_generator.data_generators.items()}
num_reps = data_gen_args.num_synthetic_generation_repeats




## XXX:
# load models in here!
primary_parameter_load_path = os.path.join(primary_model_resume_path, '{model}.{ext}')
swap_model.load_primary_state_dict(torch.load(primary_parameter_load_path.format(model = f'swap_model', ext = 'mdl'), map_location='cuda'))
for loading_smk in submodel_keys_across_set_sizes:
    submodel_resume_path = data_gen_args.submodel_resume_path_template.format(**{separation_metadata_name: loading_smk})
    submodel_state_dict: Dict[str, _T] = torch.load(os.path.join(submodel_resume_path, 'swap_model.mdl'))
    submodel_state_dict = {k: v/3. if k.endswith('S_uu_log_chol') else v for k, v in submodel_state_dict.items()}       # Because of instabilities when exping!
    swap_model.load_submodel_state_dict(loading_smk, submodel_state_dict)


swap_model.cuda()

data_output_path = data_gen_args.data_output_path
try:
    os.mkdir(data_output_path)
except FileExistsError:
    pass
data_destination = os.path.join(data_output_path, f'synthetic_data_hierarchical.npy')
config_yaml_destination = os.path.join(data_output_path, f'synthetic_data_args.yaml')
training_yaml_destination = os.path.join(data_output_path, f'synthetic_data_training_args.yaml')



# axis at front for num reps usually
all_primary_mean_function_eval = {N: np.zeros([num_reps, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}
all_submodel_mean_function_eval = {N: np.zeros([num_reps, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}
all_total_mean_function_eval = {N: np.zeros([num_reps, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}
all_primary_std_function_eval = {N: np.zeros([num_reps, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}
all_submodel_std_function_eval = {N: np.zeros([num_reps, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}
all_total_std_function_eval = {N: np.zeros([num_reps, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}
all_synthetic_prior_vectors = {N: np.zeros([num_reps, dataset_sizes[N], N + 1]).astype(float) for N in all_set_sizes}
all_synthetic_components = {N: np.zeros([num_reps, dataset_sizes[N]]) for N in all_set_sizes}
all_synthetic_estimate_errors = {N: np.zeros([num_reps, dataset_sizes[N], N]).astype(float) for N in all_set_sizes}

all_inmodel_total_log_likelihoods = {N: np.zeros([num_reps, dataset_sizes[N]]).astype(float) for N in all_set_sizes}

with torch.no_grad():

    for rep_idx in tqdm(range(num_reps)):

        for submodel_key in submodel_keys_across_set_sizes:

            for i, (set_size, dg) in enumerate(dataset_generator.data_generators.items()):

                all_smk_indices = dg.all_metadata_inverted[separation_metadata_name][submodel_key]
                all_submodel_relevant_deltas = dg.all_deltas[all_smk_indices][...,delta_dimensions].unsqueeze(0).repeat(swap_model.num_models, 1, 1, 1).to('cuda')    # [Q, M_s, N, D]
                all_submodel_errors = dg.all_errors[:,all_smk_indices,:].to('cuda')
                submodel_component_estimate_deltas = dg.all_deltas[all_smk_indices,:,1].cuda()

                minibatched_primary_inference_info = swap_model.minibatched_inference_primary(
                    all_submodel_relevant_deltas, 0, False
                )
                sample_information = swap_model.minibatched_inference_submodel(
                    submodel_key, all_submodel_relevant_deltas, minibatched_primary_inference_info.mus, minibatched_primary_inference_info.sigmas,
                    minibatched_primary_inference_info.k_uds, minibatched_primary_inference_info.K_uu_inv,
                    0, True
                )

                new_primary_mean_function_eval: _T = minibatched_primary_inference_info.mean_surface      # [1, M, N]
                new_primary_std_function_eval: _T = minibatched_primary_inference_info.std_surface        # [1, M, N]
                new_submodel_mean_function_eval: _T = sample_information.mean_submodel_surface      # [1, M, N]
                new_submodel_std_function_eval: _T = sample_information.std_submodel_surface        # [1, M, N]
                new_total_mean_function_eval: _T = sample_information.mean_total_surface     # [1, M, N]
                new_total_std_function_eval: _T = sample_information.std_total_surface      # [1, M, N]

                new_f_samples = sample_information.submodel_f_samples                  # [1, I, M, N]

                kwargs_for_generate_pi_vectors = dict(model_evaulations = new_f_samples, make_spike_and_slab = False)
                new_synth_data_batch: Dict[str, _T] = swap_model.get_generative_submodel(submodel_key).full_data_generation(set_size = set_size, vm_means = submodel_component_estimate_deltas, kwargs_for_generate_pi_vectors = kwargs_for_generate_pi_vectors)

                synthetic_errors: _T = rectify_angles(new_synth_data_batch['samples'].squeeze(0).unsqueeze(-1) - submodel_component_estimate_deltas)
                synthetic_errors_2: _T = rectify_angles(new_synth_data_batch['samples'].squeeze(0).unsqueeze(-1) + dg.all_target_zetas[all_smk_indices,0].cuda() - dg.all_target_zetas[all_smk_indices].squeeze(-1).cuda())
                assert torch.isclose(synthetic_errors, synthetic_errors_2).all()

                all_primary_mean_function_eval[set_size][rep_idx,all_smk_indices] = new_primary_mean_function_eval.cpu().numpy()
                all_primary_std_function_eval[set_size][rep_idx,all_smk_indices] = new_primary_std_function_eval.cpu().numpy()
                all_total_mean_function_eval[set_size][rep_idx,all_smk_indices] = new_total_mean_function_eval.cpu().numpy()
                all_submodel_mean_function_eval[set_size][rep_idx,all_smk_indices] = new_submodel_mean_function_eval.cpu().numpy()
                all_submodel_std_function_eval[set_size][rep_idx,all_smk_indices] = new_submodel_std_function_eval.cpu().numpy()
                all_total_std_function_eval[set_size][rep_idx,all_smk_indices] = new_total_std_function_eval.cpu().numpy()
                all_synthetic_prior_vectors[set_size][rep_idx,all_smk_indices] = new_synth_data_batch['pi_vectors'].squeeze(0).cpu().numpy()
                all_synthetic_components[set_size][rep_idx,all_smk_indices] = new_synth_data_batch['betas'].squeeze(0).cpu().numpy()
                all_synthetic_estimate_errors[set_size][rep_idx,all_smk_indices] = synthetic_errors.cpu().numpy()

                elbo_information = swap_model.get_elbo_terms(
                    submodel_key = submodel_key, deltas = all_submodel_relevant_deltas,
                    data = synthetic_errors[None], max_variational_batch_size = 0
                )

                all_inmodel_total_log_likelihoods[set_size][rep_idx,all_smk_indices] = elbo_information['likelihood_per_datapoint'].cpu().numpy()
                if elbo_information['likelihood_per_datapoint'].isnan().any():
                    import pdb; pdb.set_trace()


np.save(
    data_destination,
    {
        'generated_data': {
            "function_eval_primary_mean": all_primary_mean_function_eval,
            "function_eval_primary_std": all_primary_std_function_eval,
            "function_eval_total_mean": all_total_mean_function_eval,
            "function_eval_total_std": all_total_std_function_eval,
            "function_eval_submodel_mean": all_submodel_mean_function_eval,
            "function_eval_submodel_std": all_submodel_std_function_eval,
            "component_priors": all_synthetic_prior_vectors,
            "components": all_synthetic_components,
            "errors": all_synthetic_estimate_errors,
        },
        'inmodel_loglikelihood': all_inmodel_total_log_likelihoods
    }
)

args.write_to_yaml(config_yaml_destination)

training_yaml_info = {
    'dataset_name': 'cross_model_fit',
    'underlying_dataset': args.dataset_name,
    'synthetic_data_root': str(Path(data_output_path).parent.absolute()),
    'synthetic_data_code': 'hierarchical',
}

ConfigNamepace(training_yaml_info).write_to_yaml(training_yaml_destination)

