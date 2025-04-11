# Get laplacian approximation from synthetic data from old results...
export model_name=$1            # e.g. full_von_mises_and_uniform
export data_name=$2             # e.g. mcmaster2022_e1_oricue_cue_AR2_drop_bad_subjects
export model_idx=$3      # e.g. 0
export variational_parameterisation=$4      # e.g. gaussian
export cuda_device=$5                       # e.g. 1

export CUDA_VISIBLE_DEVICES=$cuda_device

export pypath="/homes/pr450/anaconda3/bin/python3"


echo model_name $model_name 
echo data_name  $data_name 
echo model_idx  $model_idx 



export cmd_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/j_different_error_emissions/rebuttal/laplacian_approx.py"
export hierarchical_config_yaml="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/h_hierarchical_tests/hierarchical_config_${variational_parameterisation}_multimodel.yaml"
export synthetic_data_path="/homes/pr450/repos/research_projects/error_modelling_torus/results_link_2025/j_different_error_emissions/synthetic_data/${data_name}/${model_name}_${variational_parameterisation}"


export model_resume_path="/homes/pr450/repos/research_projects/error_modelling_torus/results_link_2025/j_different_error_emissions/trained_on_data/${data_name}/${model_name}_0"





export cmd="python ${cmd_path} \
    --resume_path   ${model_resume_path}
    --hierarchical_config_yaml ${hierarchical_config_yaml}
    --data_output_path ${synthetic_data_path}
    --model_idx ${model_idx}
"

$cmd


# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh full_von_mises_and_uniform mcmaster2022_e2_dircue_lowC 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh full_von_mises_and_uniform mcmaster2022_e2_dircue_highC 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh est_dim_von_mises_and_uniform mcmaster2022_e2_dircue_lowC 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh est_dim_von_mises_and_uniform mcmaster2022_e2_dircue_highC 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh cue_dim_von_mises_and_uniform mcmaster2022_e2_dircue_lowC 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh cue_dim_von_mises_and_uniform mcmaster2022_e2_dircue_highC 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh no_dim_von_mises_and_uniform mcmaster2022_e2_dircue_lowC 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh no_dim_von_mises_and_uniform mcmaster2022_e2_dircue_highC 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh full_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR1 0 gaussian 1
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh full_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR3 0 gaussian 1
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh est_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR1 0 gaussian 1
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh est_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR3 0 gaussian 1
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR1 0 gaussian 1
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR3 0 gaussian 1
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh no_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR1 0 gaussian 1
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh no_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR3 0 gaussian 1



# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh full_von_mises_and_uniform schneegans2017_e2_cueOrientation_reportColor 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh est_dim_von_mises_and_uniform schneegans2017_e2_cueOrientation_reportColor 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh cue_dim_von_mises_and_uniform schneegans2017_e2_cueOrientation_reportColor 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh full_wrapped_stable schneegans2017_e2_cueOrientation_reportColor 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh est_dim_wrapped_stable schneegans2017_e2_cueOrientation_reportColor 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh cue_dim_wrapped_stable schneegans2017_e2_cueOrientation_reportColor 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh full_von_mises_and_uniform schneegans2017_e2_cueColor_reportOrientation 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh est_dim_von_mises_and_uniform schneegans2017_e2_cueColor_reportOrientation 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh cue_dim_von_mises_and_uniform schneegans2017_e2_cueColor_reportOrientation 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh full_wrapped_stable schneegans2017_e2_cueColor_reportOrientation 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh est_dim_wrapped_stable schneegans2017_e2_cueColor_reportOrientation 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh cue_dim_wrapped_stable schneegans2017_e2_cueColor_reportOrientation 0 gaussian 0


# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh no_dim_von_mises_and_uniform mcmaster2022_e2_dircue_medC 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh no_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh no_dim_von_mises_and_uniform schneegans2017_e2_cueOrientation_reportColor 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh no_dim_wrapped_stable schneegans2017_e2_cueOrientation_reportColor 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh no_dim_von_mises_and_uniform schneegans2017_e2_cueColor_reportOrientation 0 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/rebuttal/main_laplacian.sh no_dim_wrapped_stable schneegans2017_e2_cueColor_reportOrientation 0 gaussian 0
