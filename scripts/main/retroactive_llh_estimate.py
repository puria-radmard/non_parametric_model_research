import os, torch, numpy as np, sys
from purias_utils.util.arguments_yaml import ConfigNamepace
from purias_utils.error_modelling_torus.non_parametric_error_model.setup_utils import setup_model_whole
from purias_utils.error_modelling_torus.data_utils.loading_utils import load_experimental_data


path = sys.argv[1]


result_args = ConfigNamepace.from_yaml_path(os.path.join(path, 'args.yaml'))

MINIBATCH_SIZE = result_args.M_batch_mini

dataset_generator = load_experimental_data(result_args.dataset_name, result_args.train_indices_seed, None, result_args.M_batch, result_args.M_test_per_set_size, result_args.num_models, result_args, device = 'cpu')
all_set_sizes = list(dataset_generator.data_generators.keys())

result_args.update(ConfigNamepace({
    'all_set_sizes': all_set_sizes,
    'trainable_kernel_delta': False,
    'num_variational_samples': None,
    'num_importance_sampling_samples': 2048
}))
result_args.dict['resume_path'] = path

swap_model, D, delta_dimensions = setup_model_whole(**result_args.dict, all_min_seps = None, device = 'cuda')
swap_model.cuda()

retrospective_importance_sampled_log_likelihoods = {}

for iN, set_size in enumerate(all_set_sizes):

    dg = dataset_generator.data_generators[set_size]
    all_dataset_errors = dg.all_errors.to('cuda')
    all_dataset_relevant_deltas = dg.all_deltas[...,delta_dimensions].unsqueeze(0).repeat(swap_model.num_models, 1, 1, 1).to('cuda')           # [Q, M, N, D]
    test_time_likelihood_estimates = swap_model.refined_likelihood_estimate(all_dataset_errors, all_dataset_relevant_deltas, dg.train_indices, max_variational_batch_size=MINIBATCH_SIZE) 

    retrospective_importance_sampled_log_likelihoods[set_size] = test_time_likelihood_estimates['importance_sampled_log_likelihoods']

recent_losses = np.load(os.path.join(path, "recent_losses.npy"), allow_pickle=True).item()
recent_losses['retrospective_importance_sampled_log_likelihoods'] = retrospective_importance_sampled_log_likelihoods
import pdb; pdb.set_trace()
np.save(os.path.join(path, "recent_losses.npy"), recent_losses)



