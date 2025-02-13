# Sometimes, we still get lobe-like behaviour in the est_dim dimension of the full model.
# see for example: '/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_single_subject_summaries_7_10_24/mcmaster2022_e2_dircue_lowC/full_wrapped_stable_7/function_surface_450_4.png'

# Here, we generate from a cue-dim only model then fit it with a full model, to see if these lobes still appear

export default_config_path="/homes/pr450/repos/all_utils/purias_utils/error_modelling_torus/non_parametric_error_model/default_bnp_and_training_params.yaml"



export fitting_model_name=$1
export generating_model_name=$2
export data_name=$3
export num_subjects=$4
export cuda_device=$5
export skip_training=$6

export pypath="/homes/pr450/anaconda3/bin/python3"

export CUDA_VISIBLE_DEVICES=$cuda_device

echo fitting_model_name $fitting_model_name
echo generating_model_name $generating_model_name
echo data_name $data_name
echo num_subjects $num_subjects
echo cuda_device $cuda_device
echo skip_training $skip_training
echo
echo


# Data-model combination that generates the data
export generating_model_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/configs/model_base_configs/${generating_model_name}.yaml"
export generating_run_name_base=`cat $generating_model_config_path | sed -n -e 's/^.*run_name: //p' | sed -e 's|["",]||g'`
if [ "$num_subjects" -eq "0" ]; then
    export generating_model_base="/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_summaries_7_10_24/${data_name}"
else
    export generating_model_base="/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_single_subject_summaries_7_10_24/${data_name}"
fi

# Copy over data config and change to make it a cross_model_fit one
export generating_data_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/configs/data_base_configs/${data_name}.yaml"
export cross_fit_data_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/configs/cross_fit_data_configs/${data_name}_${generating_model_name}_${num_subjects}subjs.yaml"

export actual_data_name=`cat $generating_data_config_path | sed -n -e 's/^.*dataset_name: //p' | sed -e 's|["",]||g'`
rm $cross_fit_data_config_path
cat $generating_data_config_path | \
    sed -e "s|dataset_name:.*|dataset_name: cross_model_fit\nunderlying_dataset: "$actual_data_name"\nsynthetic_data_code: single_subject_cross_fit\nsynthetic_data_root: "$generating_model_base"\n |g" \
    >> $cross_fit_data_config_path


# This remains the same but with a different variable name
export model_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/configs/model_base_configs/${fitting_model_name}.yaml"


if [ "$num_subjects" -eq "0" ]; then

    # Generate synthetic data for *all* subject and with this data
    $pypath -m non_parametric_model.scripts.synthetic_data_generation.full_distribution \
        --resume_path ${generating_model_base}/${generating_run_name_base}_0 \
        --num_synthetic_generation_repeats 16 \
        --synthetic_data_code single_subject_cross_fit \
        --allow_model_drop

    # Define all configs
    export logging_base_root="/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_subject_aggregated_crossfits_8_10_24/${data_name}"
    # export subject_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/configs/subject_base_configs/aggregated_subjects.yaml"

    # Train this crossfit
    if [ $skip_training == "true" ]; then
        echo "not training!"
    else
        $pypath -m non_parametric_model.scripts.main.training -eap $default_config_path $model_config_path $cross_fit_data_config_path $subject_config_path \
            --train_indices_seed 0 --logging_base ${logging_base_root}/fit_to_${generating_model_name} --synthetic_data_runname ${generating_run_name_base}_0
    fi

    exit 1

else
    for sidx in $(seq 0 $(($num_subjects - 1)) );
    do  
        # Generate synthetic data for this subject and with this data
        $pypath -m non_parametric_model.scripts.synthetic_data_generation.full_distribution \
            --resume_path ${generating_model_base}/${generating_run_name_base}_${sidx} \
            --num_synthetic_generation_repeats 16 \
            --synthetic_data_code single_subject_cross_fit \
            --allow_model_drop

        # Define all configs
        export subject_config_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/configs/subject_base_configs/subject${sidx}.yaml"
        export logging_base_root="/homes/pr450/repos/research_projects/error_modelling_torus/results_link/fc_single_subject_crossfits_8_10_24/${data_name}"

        # Train this crossfit
        if [ $skip_training == "true" ]; then
            echo "not training!"
        else
            $pypath -m non_parametric_model.scripts.main.training -eap $default_config_path $model_config_path $cross_fit_data_config_path $subject_config_path \
                --train_indices_seed 0 --logging_base ${logging_base_root}/fit_to_${generating_model_name} --synthetic_data_runname ${generating_run_name_base}_${sidx}
        fi

    done
fi



# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/synth_data_script.sh full_wrapped_stable est_dim_wrapped_stable mcmaster2022_e1_oricue_cue_AR2 10 0 true
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/synth_data_script.sh est_dim_wrapped_stable est_dim_wrapped_stable mcmaster2022_e1_oricue_cue_AR2 10 0 true
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/synth_data_script.sh est_dim_wrapped_stable cue_dim_wrapped_stable mcmaster2022_e1_oricue_cue_AR2 10 0 true
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/synth_data_script.sh est_dim_wrapped_stable full_wrapped_stable mcmaster2022_e1_oricue_cue_AR2 10 0 true

# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/synth_data_script.sh full_wrapped_stable cue_dim_wrapped_stable mcmaster2022_e1_oricue_cue_AR2 10 1 true
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/synth_data_script.sh cue_dim_wrapped_stable est_dim_wrapped_stable mcmaster2022_e1_oricue_cue_AR2 10 1 true
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/synth_data_script.sh cue_dim_wrapped_stable full_wrapped_stable mcmaster2022_e1_oricue_cue_AR2 10 1 true
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/synth_data_script.sh cue_dim_wrapped_stable cue_dim_wrapped_stable mcmaster2022_e1_oricue_cue_AR2 10 1 true
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/synth_data_script.sh full_wrapped_stable full_wrapped_stable mcmaster2022_e1_oricue_cue_AR2 10 1 true

# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/synth_data_script.sh full_von_mises_and_uniform spike_and_slab_von_mises_and_uniform  mcmaster2022_e2_dircue_medC 0 1 true

# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/fc_single_subject_summaries/synth_data_script.sh asdf spike_and_slab_wrapped_stable  mcmaster2022_e1_oricue_cue_AR1 10 1 true

