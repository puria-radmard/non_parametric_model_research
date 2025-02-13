
export model_name=$1
export data_name=$2
export synthetic_data_type=$3
export fitting_submodel_parameterisation=$4
export cuda_device=$5
export splitting_key="subject_idx"

export CUDA_VISIBLE_DEVICES=$cuda_device

export pypath="/homes/pr450/anaconda3/bin/python3"


echo model_name $model_name 
echo data_name  $data_name 
echo splitting_key   $splitting_key 



export results_base='/homes/pr450/repos/research_projects/error_modelling_torus/results_link/h_hierarchical_tests_19_11_11/model_recovery'

export default_config_path="/homes/pr450/repos/all_utils/purias_utils/error_modelling_torus/non_parametric_error_model/default_bnp_and_training_params.yaml"
export model_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/configs/model_base_configs/${model_name}.yaml"
export data_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/configs/data_base_configs/${data_name}.yaml"
export hierarchical_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/h_hierarchical_tests/hierarchical_config_${fitting_submodel_parameterisation}_multimodel.yaml"
export synthetic_data_training_args_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/results_link/h_hierarchical_tests_19_11_11/synthetic_data/hierarchical_synthetic_data_${data_name}_cue_dim_wrapped_stable_${synthetic_data_type}/synthetic_data_training_args.yaml"


export logging_base="${results_base}/${data_name}"
$pypath -m non_parametric_model.scripts.hierarchical.training \
    -eap $default_config_path $model_config_path $data_config_path $hierarchical_config_path $synthetic_data_training_args_config_path \
    --train_indices_seed 0 \
    --logging_base $logging_base \
    --synthetic_data_runname    hierarchical_synthetic_data_${data_name}_cue_dim_wrapped_stable_${synthetic_data_type}


# non_parametric_model/commands/h_hierarchical_tests/train_on_synthetic_data.sh cue_dim_wrapped_stable mcmaster2022_e1_oricue_cue_AR2 gaussian gaussian  0 
