# Generate synthetic data from old results...
export model_name=$1            # e.g. full_von_mises_and_uniform
export data_name=$2             # e.g. mcmaster2022_e1_oricue_cue_AR2_drop_bad_subjects
export variational_parameterisation=$3      # e.g. gaussian
export swap_function_multiplier=$4      # e.g. 1.5
export cuda_device=$5                       # e.g. 1
export extra_args="${@:6}"  # e.g. --swap_dimensions

export CUDA_VISIBLE_DEVICES=$cuda_device

export pypath="/homes/pr450/anaconda3/bin/python3"


echo model_name $model_name 
echo data_name  $data_name 
echo variational_parameterisation  $variational_parameterisation 
echo swap_function_multiplier  $swap_function_multiplier 



export cmd_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/scripts/synthetic_data_generation/multiple_error_emissions.py"
export hierarchical_config_yaml="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/h_hierarchical_tests/hierarchical_config_${variational_parameterisation}_multimodel.yaml"
export synthetic_data_path="/homes/pr450/repos/research_projects/error_modelling_torus/results_link_2025/j_different_error_emissions/synthetic_data_mult/${data_name}/${model_name}_${variational_parameterisation}_mult${swap_function_multiplier}"


export model_resume_path="/homes/pr450/repos/research_projects/error_modelling_torus/results_link_2025/j_different_error_emissions/trained_on_data/${data_name}/${model_name}_0"





export cmd="python ${cmd_path} \
    --resume_path   ${model_resume_path}
    --num_synthetic_generation_repeats 16
    --hierarchical_config_yaml ${hierarchical_config_yaml}
    --data_output_path ${synthetic_data_path}
    --swap_function_multiplier ${swap_function_multiplier}
    --allow_model_drop
    ${extra_args}
"

$cmd


# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh est_dim_von_mises_and_uniform mcmaster2022_e2_dircue_medC gaussian 1.5 0
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh est_dim_von_mises_and_uniform mcmaster2022_e2_dircue_medC gaussian 2.5 0
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e2_dircue_medC gaussian 1.5 0
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e2_dircue_medC gaussian 2.5 0




# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 0.5 0
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 1.0 0
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 1.5 0
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 2.0 0
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 2.5 0
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian -0.5 0
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian -1.0 0
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian -1.5 0


# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 0.5 0 --swap_dimensions
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 1.0 0 --swap_dimensions
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 1.5 0 --swap_dimensions
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 2.0 0 --swap_dimensions
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 2.5 0 --swap_dimensions
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian -0.5 0 --swap_dimensions
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian -1.0 0 --swap_dimensions
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian -1.5 0 --swap_dimensions


# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 0.5 1 --duplicate_dimension 0
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 1.0 1 --duplicate_dimension 0
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 1.5 1 --duplicate_dimension 0
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 2.0 1 --duplicate_dimension 0
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 2.5 1 --duplicate_dimension 0
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian -0.5 1 --duplicate_dimension 0
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian -1.0 1 --duplicate_dimension 0
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian -1.5 1 --duplicate_dimension 0


# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 0.5 1 --duplicate_dimension 0 --num_underlying_data_duplications 10
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 1.0 1 --duplicate_dimension 0 --num_underlying_data_duplications 10
# non_parametric_model/commands/j_different_error_emissions/generate_synthetic_data_multiplied.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 2.0 1 --duplicate_dimension 0 --num_underlying_data_duplications 10
