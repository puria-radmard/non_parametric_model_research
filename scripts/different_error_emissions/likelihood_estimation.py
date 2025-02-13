import sys
from pathlib import Path

from non_parametric_model.scripts.different_error_emissions.setup import *


assert args.resume_path is not None

logging_directory = os.path.dirname(args.resume_path)


for iN, set_size in enumerate(all_set_sizes):

    dg = dataset_generator.data_generators[set_size]

    all_likelihoods_per_datapoint = torch.zeros(num_models, dg.all_deltas.shape[0])

    for iS, testing_eek in enumerate(error_emissions_keys_across_set_sizes):

        all_eek_indices = dg.all_metadata_inverted[separation_metadata_name][testing_eek]
        all_dataset_errors = dg.all_errors[:,all_eek_indices,:].to('cuda')
        all_dataset_relevant_deltas = dg.all_deltas[all_eek_indices][...,delta_dimensions].unsqueeze(0).repeat(swap_model.num_models, 1, 1, 1).to('cuda')    # [Q, M_s, N, D]

        with torch.no_grad():
            # test_time_inference_info = swap_model.get_elbo_terms(testing_eek, all_dataset_relevant_deltas, all_dataset_errors, max_variational_batch_size=MINIBATCH_SIZE, return_kl=False)
            elbo_information = swap_model.minibatched_inference(deltas=all_dataset_relevant_deltas, max_variational_batch_size = 0, take_samples = True)
            prior_info = self.swap_function.generate_pi_vectors(set_size=set_size, **kwargs_for_generate_pi_vectors)
            

        all_likelihoods_per_datapoint[:,all_eek_indices] = test_time_inference_info['likelihood_per_datapoint'].to(all_likelihoods_per_datapoint.dtype).detach().cpu()         # each [Q, M_s]
        recent_naive_log_likelihoods[set_size][testing_eek] = test_time_inference_info['likelihood_per_datapoint'].detach().cpu().numpy()
    

recent_losses = {
    "recent_naive_log_likelihoods": recent_naive_log_likelihoods
}

np.save(os.path.join(logging_directory, "better_likelihood_estimates.npy"), recent_losses)
