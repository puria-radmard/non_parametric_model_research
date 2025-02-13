
export model_name=$1
export data_name=$2
export submodel_parameterisation=$3
export cuda_device=$4

export CUDA_VISIBLE_DEVICES=$cuda_device

export pypath="/homes/pr450/anaconda3/bin/python3"


echo model_name $model_name 
echo data_name  $data_name 



export results_base='/homes/pr450/repos/research_projects/error_modelling_torus/results_link_2025/j_different_error_emissions/trained_on_data'


export default_config_path="/homes/pr450/repos/all_utils/purias_utils/error_modelling_torus/non_parametric_error_model/default_bnp_and_training_params.yaml"
export model_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/all_configs/model_base_configs/${model_name}.yaml"
export data_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/all_configs/data_base_configs/${data_name}.yaml"
export hierarchical_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/h_hierarchical_tests/hierarchical_config_${submodel_parameterisation}_multimodel.yaml"


export logging_base="${results_base}/${data_name}"
$pypath -m non_parametric_model.scripts.different_error_emissions.training \
    -eap $default_config_path $model_config_path $data_config_path $hierarchical_config_path \
    --train_indices_seed 0 \
    --logging_base $logging_base


# non_parametric_model/commands/j_different_error_emissions/main.sh full_von_mises_and_uniform mcmaster2022_e2_dircue_medC_drop_bad_subjects gaussian 0
# non_parametric_model/commands/j_different_error_emissions/main.sh full_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2_drop_bad_subjects gaussian 0
# non_parametric_model/commands/j_different_error_emissions/main.sh cue_dim_von_mises_and_uniform mcmaster2022_e2_dircue_medC_drop_bad_subjects gaussian 0
# non_parametric_model/commands/j_different_error_emissions/main.sh est_dim_von_mises_and_uniform mcmaster2022_e2_dircue_medC_drop_bad_subjects gaussian 0




# non_parametric_model/commands/j_different_error_emissions/main.sh full_von_mises_and_uniform mcmaster2022_e2_dircue_medC gaussian 0
# non_parametric_model/commands/j_different_error_emissions/main.sh full_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/main.sh cue_dim_von_mises_and_uniform mcmaster2022_e2_dircue_medC gaussian 1
# non_parametric_model/commands/j_different_error_emissions/main.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 1
# non_parametric_model/commands/j_different_error_emissions/main.sh est_dim_von_mises_and_uniform mcmaster2022_e2_dircue_medC gaussian 1
# non_parametric_model/commands/j_different_error_emissions/main.sh est_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/main.sh no_dim_von_mises_and_uniform mcmaster2022_e2_dircue_medC gaussian 0
# non_parametric_model/commands/j_different_error_emissions/main.sh no_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR2 gaussian 0



# non_parametric_model/commands/j_different_error_emissions/main.sh est_dim_von_mises_and_uniform schneegans2017_e2_cueOrientation_reportColor gaussian 0
# non_parametric_model/commands/j_different_error_emissions/main.sh est_dim_von_mises_and_uniform schneegans2017_e2_cueColor_reportOrientation gaussian 1
# non_parametric_model/commands/j_different_error_emissions/main.sh cue_dim_von_mises_and_uniform schneegans2017_e2_cueOrientation_reportColor gaussian 0
# non_parametric_model/commands/j_different_error_emissions/main.sh cue_dim_von_mises_and_uniform schneegans2017_e2_cueColor_reportOrientation gaussian 1
# non_parametric_model/commands/j_different_error_emissions/main.sh full_von_mises_and_uniform schneegans2017_e2_cueOrientation_reportColor gaussian 0
# non_parametric_model/commands/j_different_error_emissions/main.sh full_von_mises_and_uniform schneegans2017_e2_cueColor_reportOrientation gaussian 1
# non_parametric_model/commands/j_different_error_emissions/main.sh no_dim_von_mises_and_uniform schneegans2017_e2_cueOrientation_reportColor gaussian 0
# non_parametric_model/commands/j_different_error_emissions/main.sh no_dim_von_mises_and_uniform schneegans2017_e2_cueColor_reportOrientation gaussian 1
# non_parametric_model/commands/j_different_error_emissions/main.sh est_dim_wrapped_stable schneegans2017_e2_cueOrientation_reportColor gaussian 0
# non_parametric_model/commands/j_different_error_emissions/main.sh est_dim_wrapped_stable schneegans2017_e2_cueColor_reportOrientation gaussian 1
# non_parametric_model/commands/j_different_error_emissions/main.sh cue_dim_wrapped_stable schneegans2017_e2_cueOrientation_reportColor gaussian 0
# non_parametric_model/commands/j_different_error_emissions/main.sh cue_dim_wrapped_stable schneegans2017_e2_cueColor_reportOrientation gaussian 1
# non_parametric_model/commands/j_different_error_emissions/main.sh full_wrapped_stable schneegans2017_e2_cueOrientation_reportColor gaussian 0
# non_parametric_model/commands/j_different_error_emissions/main.sh full_wrapped_stable schneegans2017_e2_cueColor_reportOrientation gaussian 1
# non_parametric_model/commands/j_different_error_emissions/main.sh no_dim_wrapped_stable schneegans2017_e2_cueOrientation_reportColor gaussian 0
# non_parametric_model/commands/j_different_error_emissions/main.sh no_dim_wrapped_stable schneegans2017_e2_cueColor_reportOrientation gaussian 1

####### XXX ####### XXX ####### XXX ####### XXX ####### XXX ####### XXX ####### XXX ####### XXX ####### XXX ####### XXX 

# non_parametric_model/commands/j_different_error_emissions/main.sh full_von_mises_and_uniform mcmaster2022_e2_dircue_highC gaussian 0
# non_parametric_model/commands/j_different_error_emissions/main.sh no_dim_von_mises_and_uniform mcmaster2022_e2_dircue_highC gaussian 0
# non_parametric_model/commands/j_different_error_emissions/main.sh cue_dim_von_mises_and_uniform mcmaster2022_e2_dircue_highC gaussian 1
# non_parametric_model/commands/j_different_error_emissions/main.sh est_dim_von_mises_and_uniform mcmaster2022_e2_dircue_highC gaussian 1

# non_parametric_model/commands/j_different_error_emissions/main.sh cue_dim_von_mises_and_uniform mcmaster2022_e2_dircue_lowC gaussian 1
# non_parametric_model/commands/j_different_error_emissions/main.sh est_dim_von_mises_and_uniform mcmaster2022_e2_dircue_lowC gaussian 1
# non_parametric_model/commands/j_different_error_emissions/main.sh full_von_mises_and_uniform mcmaster2022_e2_dircue_lowC gaussian 0
# non_parametric_model/commands/j_different_error_emissions/main.sh no_dim_von_mises_and_uniform mcmaster2022_e2_dircue_lowC gaussian 0



# non_parametric_model/commands/j_different_error_emissions/main.sh full_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR1 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/main.sh no_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR1 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/main.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR1 gaussian 1
# non_parametric_model/commands/j_different_error_emissions/main.sh est_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR1 gaussian 1

# non_parametric_model/commands/j_different_error_emissions/main.sh cue_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR3 gaussian 1
# non_parametric_model/commands/j_different_error_emissions/main.sh est_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR3 gaussian 1
# non_parametric_model/commands/j_different_error_emissions/main.sh full_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR3 gaussian 0
# non_parametric_model/commands/j_different_error_emissions/main.sh no_dim_von_mises_and_uniform mcmaster2022_e1_oricue_cue_AR3 gaussian 0



