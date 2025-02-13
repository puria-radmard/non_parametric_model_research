# Generate synthetic data from old results...

export submodel_type=$1 # gaussian or vanilla

export cmd_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/scripts/synthetic_data_generation/hierarchical.py"
export synthetic_data_path="/homes/pr450/repos/research_projects/error_modelling_torus/results_link/h_hierarchical_tests_19_11_11/synthetic_data/hierarchical_synthetic_data_mcmaster2022_e1_oricue_cue_AR2_cue_dim_wrapped_stable_${submodel_type}"

export primary_model_resume_path="/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_summaries_7_10_24/mcmaster2022_e1_oricue_cue_AR2/cue_dim_wrapped_stable_0"
export hierarchical_config_yaml="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/h_hierarchical_tests/hierarchical_config_${submodel_type}_multimodel.yaml"
export submodel_resume_path_template="/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_single_subject_summaries_7_10_24/mcmaster2022_e1_oricue_cue_AR2/cue_dim_wrapped_stable_{subject_idx}"


export cmd="python ${cmd_path} \
    --primary_model_resume_path ${primary_model_resume_path}    \
    --hierarchical_config_yaml ${hierarchical_config_yaml}     \
    --submodel_resume_path_template ${submodel_resume_path_template}    \
    --data_output_path ${synthetic_data_path}     \
    --num_synthetic_generation_repeats 16   \
    --allow_model_drop
"

$cmd
