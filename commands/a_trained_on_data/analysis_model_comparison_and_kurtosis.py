import os, json, torch
from glob import glob
from purias_utils.util.arguments_yaml import ConfigNamepace

from purias_utils.error_modelling_torus.data_utils.loading_utils import load_experimental_data
from purias_utils.error_modelling_torus.non_parametric_error_model.setup_utils import setup_model_whole

from non_parametric_model.commands.a_trained_on_data.kurtosis_utils import get_emissions_kurtosis, errors_kurtosis_values, residuals_kurtosis_values, generate_full_residual_particle_set

dataset_names = None# ['bays2009']
do_kurtosis = False
do_fitted_kurtosis = False

config_base_path = "/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data"
all_yaml_paths = glob(os.path.join(config_base_path, "*.yaml"))
all_yamls = [ConfigNamepace.from_yaml_path(yaml_path) for yaml_path in all_yaml_paths]

if dataset_names is None:
    dataset_names = list(set([x.dataset_name for x in all_yamls]))
else:
    dataset_names = dataset_names

for ds_name in dataset_names:

    all_dataset_yamls = list(filter(lambda x: x.dataset_name == ds_name, all_yamls))
    all_logging_bases = [cfg.logging_base for cfg in all_dataset_yamls]

    if do_kurtosis:
        column_names = [
            'model_class','N','training_set_seed','raw_kurtosis_fitted', 'raw_kurtosis_particle', 'emissions_kurtosis', 'residual_kurtosis_fitted', 'residual_kurtosis_particle', 'test_llh','test_sizes','train_llh','train_sizes',
        ]
    else:
        column_names = [
            'model_class','N','training_set_seed','test_llh','test_sizes','train_llh','train_sizes',
        ]

    results_dest = f'/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/figures/source_tables/model_comparison_and_kurtosis_{ds_name}.txt'

    with open(results_dest, 'w') as f:
        for col in column_names:
            f.write(str(col))
            f.write(',')
        f.write('\n')

    for logging_base in all_logging_bases:
        
        model_name = logging_base.split('/')[-1]

        for run_path in glob(os.path.join(logging_base, "run_*")):
            
            run_args_path = os.path.join(run_path, "args.yaml")
            run_args = ConfigNamepace.from_yaml_path(run_args_path)
            training_set_seed = run_args.train_indices_seed

            dataset_generator = load_experimental_data(run_args.dataset_name, run_args.train_indices_seed, run_args.train_indices_path, run_args.M_batch, run_args.M_test_per_set_size, run_args)
            all_set_sizes = list(dataset_generator.data_generators.keys())

            generative_model, variational_models, variational_model, D, delta_dimensions = setup_model_whole(
                run_args.swap_type, run_args.emission_type, 
                all_set_sizes, run_args.remove_uniform, run_args.include_pi_u_tilde, False, 
                run_args.R_per_dim, run_args.fix_non_swap, run_args.include_pi_1_tilde,
                run_args.fix_inducing_point_locations, 
                run_args.shared_swap_function, run_args.shared_emission_distribution,
                device = 'cuda'
            )

            parameter_load_path = os.path.join(run_path, '{model}.{ext}')
            generative_model.load_state_dict(torch.load(parameter_load_path.format(model = f'generative_model', ext = 'mdl')))
            for k, v in variational_models.items():
                v.load_state_dict(torch.load(parameter_load_path.format(model = f'variational_model_{k}', ext = 'mdl')))

            with open(os.path.join(run_path, f"train_indices.json"), 'r') as jf:
                train_indices = json.load(jf)
            with open(os.path.join(run_path, f"recent_losses.json"), 'r') as jf:
                recent_losses = json.load(jf)

            for N, dg in dataset_generator.data_generators.items():

                result_row = [
                        model_name,
                        N,
                        training_set_seed,
                ]
                
                if do_kurtosis:

                    if not run_args.shared_swap_function:
                        variational_model = variational_models[N]
                    
                    import pdb; pdb.set_trace(header = 'loading in train indices should not be needed: check')
                    dg.set_train_indices(train_indices[str(N)]) # redundant
                    all_errors = dg.all_errors[train_indices[str(N)]].detach()
                    all_deltas = dg.all_deltas[train_indices[str(N)]].detach()
                    all_relevant_deltas = all_deltas[...,delta_dimensions]

                    (
                        particle_locations,
                        particle_weights_non_uniform,
                        particle_weights_uniform
                    ) = generate_full_residual_particle_set(generative_model, variational_model, run_args.swap_type, all_relevant_deltas, all_errors, num_gp_samples = 128)
                    residual_knot_weights = particle_weights_non_uniform + particle_weights_uniform

                    raw_kurtosis_fitted, raw_kurtosis_particle =  errors_kurtosis_values(all_errors, num_iter=150, num_fits=1, do_fitted_kurtosis=do_fitted_kurtosis)
                    emissions_kurtosis = get_emissions_kurtosis(N, run_args.emission_type, generative_model)
                    residual_kurtosis_fitted, residual_kurtosis_particle = residuals_kurtosis_values(particle_locations, residual_knot_weights, num_iter=150, num_fits=1, do_fitted_kurtosis=do_fitted_kurtosis)

                    result_row.extend([
                        raw_kurtosis_fitted,
                        raw_kurtosis_particle, 
                        emissions_kurtosis, 
                        residual_kurtosis_fitted,
                        residual_kurtosis_particle
                    ])
                    
                result_row.extend([
                    recent_losses['recent_average_test_llh'][str(N)],
                    recent_losses['recent_test_sizes'][str(N)],
                    recent_losses['recent_average_train_llh'][str(N)],
                    recent_losses['recent_train_sizes'][str(N)],
                ])
                    


                with open(results_dest, 'a') as f:
                    for res in result_row:
                        f.write(str(res))
                        f.write(',')
                    f.write('\n')