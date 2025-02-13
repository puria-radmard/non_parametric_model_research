
export fitting_model_name=$1
export fitting_model_variational_parameterisation=$2
export generating_model_name=$3
export generating_model_variational_parameterisation=$4
export data_name=$5
export cuda_device=$6

export CUDA_VISIBLE_DEVICES=$cuda_device

export pypath="/homes/pr450/anaconda3/bin/python3"


echo fitting_model_name $fitting_model_name
echo fitting_model_variational_parameterisation $fitting_model_variational_parameterisation
echo generating_model_name $generating_model_name
echo generating_model_variational_parameterisation $generating_model_variational_parameterisation
echo data_name $data_name
echo cuda_device $cuda_device


export results_base="/homes/pr450/repos/research_projects/error_modelling_torus/results_link_2025/j_different_error_emissions/trained_on_synthetic_data_duplicate_dimensions/fit_to_${generating_model_name}_${generating_model_variational_parameterisation}_duplicate_dimensions"
export default_config_path="/homes/pr450/repos/all_utils/purias_utils/error_modelling_torus/non_parametric_error_model/default_bnp_and_training_params.yaml"
export model_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/all_configs/model_base_configs/${fitting_model_name}.yaml"
export data_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/all_configs/data_base_configs/${data_name}.yaml"
export hierarchical_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/h_hierarchical_tests/hierarchical_config_${fitting_model_variational_parameterisation}_multimodel.yaml"


export synthetic_data_training_args_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/results_link_2025/j_different_error_emissions/synthetic_data_duplicate_dimensions/${data_name}/${generating_model_name}_${generating_model_variational_parameterisation}_duplicate_dimensions/synthetic_data_training_args.yaml"


export logging_base="${results_base}/${data_name}"
$pypath -m non_parametric_model.scripts.different_error_emissions.training \
    -eap $default_config_path $model_config_path $data_config_path $hierarchical_config_path $synthetic_data_training_args_config_path \
    --train_indices_seed 0 \
    --logging_base $logging_base \
    --synthetic_data_runname    ${generating_model_name}_${generating_model_variational_parameterisation}_duplicate_dimensions




# non_parametric_model/commands/j_different_error_emissions/train_on_synthetic_data_duplicate_dimensions.sh cue_dim_von_mises_and_uniform gaussian cue_dim_von_mises_and_uniform gaussian mcmaster2022_e1_oricue_cue_AR2 1
# non_parametric_model/commands/j_different_error_emissions/train_on_synthetic_data_duplicate_dimensions.sh est_dim_von_mises_and_uniform gaussian cue_dim_von_mises_and_uniform gaussian mcmaster2022_e1_oricue_cue_AR2 1
# non_parametric_model/commands/j_different_error_emissions/train_on_synthetic_data_duplicate_dimensions.sh full_von_mises_and_uniform gaussian cue_dim_von_mises_and_uniform gaussian mcmaster2022_e1_oricue_cue_AR2 0

