# See g_underfitting_checks README

export default_config_path="/homes/pr450/repos/all_utils/purias_utils/error_modelling_torus/non_parametric_error_model/default_bnp_and_training_params.yaml"


export fitting_model_name=$1
export data_name=$2
export cuda_device=$3

export generating_model_name="full_wrapped_stable"
export num_subjects="10"
export drop_size="864"

export pypath="/homes/pr450/anaconda3/bin/python3"

export CUDA_VISIBLE_DEVICES=$cuda_device

echo fitting_model_name $fitting_model_name
echo data_name $data_name
echo cuda_device $cuda_device
echo



# Data-model combination that generates the data
export generating_model_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/configs/model_base_configs/${generating_model_name}.yaml"
export generating_run_name_base=`cat $generating_model_config_path | sed -n -e 's/^.*run_name: //p' | sed -e 's|["",]||g'`
export generating_model_base="/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_summaries_7_10_24/${data_name}" # Always using data generated from model trained on all the subjects' data

# Copy over data config and change to make it a cross_model_fit one
export generating_data_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/configs/data_base_configs/${data_name}.yaml"
export cross_fit_data_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/g_underfitting_checks/cross_fit_data_configs/${data_name}_${generating_model_name}_${num_subjects}subjs_reduced.yaml"
export cross_fit_data_config_path_no_discard="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/g_underfitting_checks/cross_fit_data_configs/${data_name}_${generating_model_name}_${num_subjects}subjs_full.yaml"
export actual_data_name=`cat $generating_data_config_path | sed -n -e 's/^.*dataset_name: //p' | sed -e 's|["",]||g'`
rm $cross_fit_data_config_path
rm $cross_fit_data_config_path_no_discard
cat $generating_data_config_path | \
    sed -e "s|dataset_name:.*|num_training_examples: 1000\ndiscard_last_n_training_examples: "$drop_size"\ndataset_name: cross_model_fit\nunderlying_dataset: "$actual_data_name"\nsynthetic_data_code: single_subject_cross_fit\nsynthetic_data_root: "$generating_model_base"\n |g" \
    >> $cross_fit_data_config_path
cat $generating_data_config_path | \
    sed -e "s|dataset_name:.*|num_training_examples: 1000\ndataset_name: cross_model_fit\nunderlying_dataset: "$actual_data_name"\nsynthetic_data_code: single_subject_cross_fit\nsynthetic_data_root: "$generating_model_base"\n |g" \
    >> $cross_fit_data_config_path_no_discard

# This remains the same but with a different variable name
export model_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/configs/model_base_configs/${fitting_model_name}.yaml"


# Start by crossfitting all data
export logging_base_root="/homes/pr450/repos/research_projects/error_modelling_torus/results_link/g_underfitting_checks/${data_name}"

# Train this crossfit
# We don't actually split by subjects, this causes problems in loading_utils
# Instead, we randomly shuffle the whole set and drop all but the first 96 items (equivalent to 1 subject remaining)
$pypath -m non_parametric_model.scripts.main.training -eap $default_config_path $model_config_path $cross_fit_data_config_path_no_discard \
    --train_indices_seed 0 --logging_base ${logging_base_root}/fit_to_${generating_model_name}_notreduced --synthetic_data_runname ${generating_run_name_base}_0


# Loop over subjects and limit training to just that one! (sort of!!!)
for sidx in $(seq 0 $(($num_subjects - 1)) );
do  

    # Define all configs
    export logging_base_root="/homes/pr450/repos/research_projects/error_modelling_torus/results_link/g_underfitting_checks/${data_name}"

    # Train this crossfit
    # We don't actually split by subjects, this causes problems in loading_utils
    # Instead, we randomly shuffle the whole set and drop all but the first 96 items (equivalent to 1 subject remaining)
    $pypath -m non_parametric_model.scripts.main.training -eap $default_config_path $model_config_path $cross_fit_data_config_path \
        --train_indices_seed $sidx --logging_base ${logging_base_root}/fit_to_${generating_model_name}_reduced --synthetic_data_runname ${generating_run_name_base}_0

done


# non_parametric_model/commands/g_underfitting_checks/single_subject_synth_data_script.sh full_wrapped_stable mcmaster2022_e2_dircue_medC 1
# non_parametric_model/commands/g_underfitting_checks/single_subject_synth_data_script.sh full_wrapped_stable mcmaster2022_e2_dircue_lowC 1
# non_parametric_model/commands/g_underfitting_checks/single_subject_synth_data_script.sh full_wrapped_stable mcmaster2022_e2_dircue_highC 1
# non_parametric_model/commands/g_underfitting_checks/single_subject_synth_data_script.sh est_dim_wrapped_stable mcmaster2022_e2_dircue_medC 1
# non_parametric_model/commands/g_underfitting_checks/single_subject_synth_data_script.sh est_dim_wrapped_stable mcmaster2022_e2_dircue_lowC 1
# non_parametric_model/commands/g_underfitting_checks/single_subject_synth_data_script.sh est_dim_wrapped_stable mcmaster2022_e2_dircue_highC 1
# non_parametric_model/commands/g_underfitting_checks/single_subject_synth_data_script.sh cue_dim_wrapped_stable mcmaster2022_e2_dircue_medC 0
# non_parametric_model/commands/g_underfitting_checks/single_subject_synth_data_script.sh cue_dim_wrapped_stable mcmaster2022_e2_dircue_lowC 0
# non_parametric_model/commands/g_underfitting_checks/single_subject_synth_data_script.sh cue_dim_wrapped_stable mcmaster2022_e2_dircue_highC 0
# non_parametric_model/commands/g_underfitting_checks/single_subject_synth_data_script.sh spike_and_slab_wrapped_stable mcmaster2022_e2_dircue_medC 0
# non_parametric_model/commands/g_underfitting_checks/single_subject_synth_data_script.sh spike_and_slab_wrapped_stable mcmaster2022_e2_dircue_lowC 0
# non_parametric_model/commands/g_underfitting_checks/single_subject_synth_data_script.sh spike_and_slab_wrapped_stable mcmaster2022_e2_dircue_highC 0
