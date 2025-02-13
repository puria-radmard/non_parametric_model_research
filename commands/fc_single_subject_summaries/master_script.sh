
export model_name=$1
export data_name=$2
export num_subjects=$3
export cuda_device=$4

export CUDA_VISIBLE_DEVICES=$cuda_device

export pypath="/homes/pr450/anaconda3/bin/python3"


echo model_name $model_name 
echo data_name  $data_name 
echo num_subjects   $num_subjects 

export default_config_path="/homes/pr450/repos/all_utils/purias_utils/error_modelling_torus/non_parametric_error_model/default_bnp_and_training_params.yaml"
export model_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/configs/model_base_configs/${model_name}.yaml"
export data_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/configs/data_base_configs/${data_name}.yaml"

if [ "$num_subjects" -eq "0" ]; then
    #export subject_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/configs/subject_base_configs/aggregated_subjects.yaml"
    #rm $subject_config_path
    #echo "num_training_examples: 2000" >> $subject_config_path
    export logging_base="/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_summaries_7_10_24/${data_name}"
    $pypath -m non_parametric_model.scripts.main.training -eap $default_config_path $model_config_path $data_config_path --train_indices_seed 0 --logging_base $logging_base
else
    export logging_base="/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_single_subject_summaries_7_10_24/${data_name}"
    for sidx in $(seq 1 $(($num_subjects - 1)) );
    do
        export subject_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/configs/subject_base_configs/subject${sidx}.yaml"
        rm $subject_config_path
        echo "subjects:
            - ${sidx}" >> $subject_config_path
        $pypath -m non_parametric_model.scripts.main.training -eap $default_config_path $model_config_path $data_config_path $subject_config_path --train_indices_seed 0 --logging_base $logging_base
    done
fi


# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/master_script.sh full_von_mises_and_uniform vandenberg2012_color 0 0
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/master_script.sh full_wrapped_stable vandenberg2012_color 10 1
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/master_script.sh est_dim_wrapped_stable vandenberg2012_color 10 1
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/master_script.sh cue_dim_wrapped_stable vandenberg2012_color 10 0
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/master_script.sh cue_dim_wrapped_stable vandenberg2012_orientation 10 0
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/master_script.sh full_wrapped_stable vandenberg2012_orientation 10 0

# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/master_script.sh est_dim_wrapped_stable vandenberg2012_orientation 10 1
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/master_script.sh spike_and_slab_wrapped_stable vandenberg2012_orientation 10 0
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/synth_data_script.sh est_dim_wrapped_stable spike_and_slab_wrapped_stable vandenberg2012_orientation 1 1 false

# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/master_script.sh est_dim_wrapped_stable vandenberg2012_orientation_allow_move 1 0

# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/master_script.sh est_dim_wrapped_stable bays2014_orientation 1 0


# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/master_script.sh cue_dim_von_mises_and_uniform mcmaster2022_e3_rand 0 1
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/master_script.sh spike_and_slab_von_mises_and_uniform mcmaster2022_e3_rand 0 0


# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/master_script.sh spike_and_slab_wrapped_stable schneegans2017_e2_cueColor_reportOrientation 0 1 
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/master_script.sh cue_dim_wrapped_stable schneegans2017_e2_cueColor_reportOrientation 0 0
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/master_script.sh est_dim_wrapped_stable schneegans2017_e2_cueColor_reportOrientation 0 0 
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/master_script.sh full_wrapped_stable schneegans2017_e2_cueColor_reportOrientation 0 1 

# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/synth_data_script.sh est_dim_wrapped_stable spike_and_slab_wrapped_stable schneegans2017_e2_cueOrientation_reportColor 0 1 false

# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/master_script.sh full_wrapped_stable mcmaster2022_e1_loccue_cue_AR1 0 1


################################################################

# TODO:

# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/master_script.sh cue_dim_wrapped_stable vandenberg2012_color 10 0
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/master_script.sh est_dim_von_mises_and_uniform vandenberg2012_color 10 0
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/master_script.sh full_von_mises_and_uniform vandenberg2012_color 10 1
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/master_script.sh spike_and_slab_wrapped_stable vandenberg2012_color 10 1


