"""
11.04.25 - rebuttal
"""

# Load up a model and its training data, then go over and generate data from an integral over the full q(f | \psi)
# This swap function posterior is the same across all the data, but we do differentiate based on 

import os, torch, argparse, json
from math import log

from torch.func import functional_call
from torch.func import jacrev, jacfwd, vmap

from purias_utils.util.arguments_yaml import ConfigNamepace
from purias_utils.error_modelling_torus.non_parametric_error_model.setup_utils import setup_model_whole, MultipleErrorEmissionsWorkingMemoryFullSwapModel
from purias_utils.error_modelling_torus.data_utils.loading_utils import load_experimental_data


SAVE_PATH = 'non_parametric_model/commands/j_different_error_emissions/rebuttal/laplacian_results.json'


parser = argparse.ArgumentParser()
parser.add_argument('--resume_path', type = str)
parser.add_argument('--model_idx', type = int)
parser.add_argument('--hierarchical_config_yaml', type = str, help="", required = False, default = "/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/h_hierarchical_tests/hierarchical_config_gaussian.yaml")
parser.add_argument('--data_output_path', type = str, help = 'Since we are not using one model, we need an output math for the data! It will include signatures back to the models used')
resuming_args = parser.parse_args()

resume_path = resuming_args.resume_path
assert resume_path is not None
args = ConfigNamepace.from_yaml_path(os.path.join(resume_path, "args.yaml"))
extra_args = ConfigNamepace.from_yaml_path(resuming_args.hierarchical_config_yaml)
args.dict.pop('resume_path')


dataset_generator = load_experimental_data(args.dataset_name, args.train_indices_seed, args.train_indices_path, args.M_batch, args.M_test_per_set_size, 1, args, 'cpu')
all_set_sizes = list(dataset_generator.data_generators.keys())

separation_metadata_name = extra_args.separation_metadata_name
error_emissions_keys_across_set_sizes = [set(dg.all_metadata_inverted[separation_metadata_name].keys()) for N, dg in dataset_generator.data_generators.items()]
error_emissions_keys_across_set_sizes = list(set.union(*error_emissions_keys_across_set_sizes))

swap_model, D, delta_dimensions = setup_model_whole(
    **args.dict, error_emissions_keys=error_emissions_keys_across_set_sizes,
    all_set_sizes=all_set_sizes, trainable_kernel_delta=False, all_min_seps=None,
    num_variational_samples=args.monte_carlo_gp_samples,
    num_importance_sampling_samples=args.monte_carlo_gp_samples, resume_path=resume_path
)
swap_model: MultipleErrorEmissionsWorkingMemoryFullSwapModel
swap_model.cuda()


print('REDUCING TO A SINGLE MODEL!')
swap_model.reduce_to_single_model(model_index=int(resuming_args.model_idx))



MINIBATCH_SIZE = args.M_batch_mini



def log_likelihood(generative_model_params, param_names, kwargs):
    new_params = {k: v for k, v in swap_model.named_parameters()}
    for n, p in zip(param_names, generative_model_params):
        assert 'generative_model.' + n in new_params
        new_params['generative_model.' + n] = p

    return functional_call(
        swap_model,
        new_params,
        args = None,
        kwargs = kwargs,
    )

def custom_hessian(func, argnums=0):
    return jacrev(jacrev(func, argnums=argnums), argnums=argnums)

# def custom_hessian(func, argnums=0, randomness='different'):
#     def first_grad(*args):
#         return jacrev(func, argnums=argnums)(*args)
    
#     def wrapped(*args):
#         # Wrap first_grad with a vmap that handles randomness
#         batched_grad = vmap(first_grad, randomness=randomness)
#         return jacrev(batched_grad, argnums=argnums)(*args)
    
#     return wrapped


hessian_func = custom_hessian(log_likelihood)


all_new_results = []


for iN, set_size in enumerate(all_set_sizes):

    dg = dataset_generator.data_generators[set_size]

    # all_likelihoods_per_datapoint = torch.zeros(swap_model.num_models, dg.all_deltas.shape[0])

    num_total_datapoints = dg.all_errors.shape[1]
    log2pi = log(2 * torch.pi)

    for iS, testing_eek in enumerate(error_emissions_keys_across_set_sizes):
        
        all_eek_indices = dg.all_metadata_inverted[separation_metadata_name][testing_eek]
        all_dataset_errors = dg.all_errors[:,all_eek_indices,:].to('cuda')
        all_dataset_relevant_deltas = dg.all_deltas[all_eek_indices][...,delta_dimensions].unsqueeze(0).repeat(swap_model.num_models, 1, 1, 1).to('cuda')    # [Q, M_s, N, D]

        # all_test_time_inference_info = swap_model.get_elbo_terms(testing_eek, all_dataset_relevant_deltas, all_dataset_errors, max_variational_batch_size=MINIBATCH_SIZE, return_kl=False)
        # all_likelihoods_per_datapoint[:,all_eek_indices] = all_test_time_inference_info['likelihood_per_datapoint'].to(all_likelihoods_per_datapoint.dtype).detach().cpu()         # each [Q, M_s]

        num_eek_datapoints = all_dataset_relevant_deltas.shape[1]

        assert swap_model.num_models == 1

        num_shared_params, num_eek_params = 0, 0
        relevant_generative_parameter_names, relevant_generative_model_parameters = [], []
        for k, v in swap_model.generative_model.named_parameters():
            if k.startswith('error_emissions'):
                if k.split('.')[1] != str(testing_eek):
                    continue
                else:
                    num_eek_params += v.numel()
                    assert num_eek_params % 1.0 == 0.0
            else:
                num_shared_params += v.numel()
                assert num_shared_params % 1.0 == 0.0
            relevant_generative_parameter_names.append(k)
            relevant_generative_model_parameters.append(v)
        
        kwargs = {
            'error_emissions_key': testing_eek,
            'deltas': all_dataset_relevant_deltas,
            'data': all_dataset_errors,
            'max_variational_batch_size': MINIBATCH_SIZE,
            'return_kl': False,
        }
        llh = log_likelihood(relevant_generative_model_parameters, relevant_generative_parameter_names, kwargs)
        hess = hessian_func(relevant_generative_model_parameters, relevant_generative_parameter_names, kwargs)
        
        num_relevant_params = int(num_shared_params + num_eek_params)
        H_matrix = torch.zeros(num_relevant_params, num_relevant_params)

        row_offset = 0
        for i, pi in enumerate(relevant_generative_model_parameters):
            pi_flat = pi.reshape(-1)
            ni = pi_flat.numel()
            col_offset = 0
            for j, pj in enumerate(relevant_generative_model_parameters):
                pj_flat = pj.reshape(-1)
                nj = pj_flat.numel()
                
                H_block = hess[i][j].reshape(ni, nj)  # flatten block
                H_matrix[row_offset:row_offset+ni, col_offset:col_offset+nj] = H_block
                
                col_offset += nj
            row_offset += ni


        det_neg_hessian = torch.linalg.det(-H_matrix)
        log_det_neg_hessian = det_neg_hessian.log().item()
        print(det_neg_hessian.item(), log_det_neg_hessian, llh.item())


        average_eek_llh = llh.item()
        eek_penalisation = 0.5 * log2pi * num_eek_params
        shared_penalisation = 0.5 * log2pi * num_shared_params
        hessian_term = - 0.5 * log_det_neg_hessian
        
        total_laplac = average_eek_llh + eek_penalisation + shared_penalisation + hessian_term
        
        result = {
            'model_idx': int(resuming_args.model_idx),
            'subject': int(testing_eek),
            'data': resume_path.split('/')[-2],
            'model': resume_path.split('/')[-1],
            'total_laplac': float(total_laplac),
        }

        all_new_results.append(result)



with open(SAVE_PATH) as f:
    existing_results = json.load(f)

existing_results.extend(all_new_results)

with open(SAVE_PATH, 'w') as f:
    json.dump(existing_results, f, indent=2)
    