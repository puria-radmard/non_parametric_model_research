# e.g. non_parametric_model/commands/d_finalise_kurtosis_results/sweep_circular_statistics_across_trained_models.sh mcmaster2022_e1_oricue_cue_dim_wrapped_stable mcmaster2022_e1_oricue_spike_and_slab_wrapped_stable

export configs_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/d_finalise_kurtosis_results/configs"

export experiment_names="${@:1}"


export logging_bases=()
export experiment_names_array=()

for experiment_name in ${experiment_names[@]}
do
    export training_yaml_path="${configs_path}/${experiment_name}.yaml"
    logging_base=`cat $training_yaml_path | sed -n -e 's/^.*logging_base: //p' | sed -e 's|["",]||g' | sed -e 's|['',]||g'`
    logging_bases+=($logging_base)
    experiment_names_array+=($experiment_name)
done






for i in $(seq 0 $((${#logging_bases[@]} - 1)));
do
    export inferring_model_base=${logging_bases[i]}

    for j in $(seq 0 $((${#logging_bases[@]} - 1)));
    do

        if [[ "$i" == "$j" ]]; then
            continue
        fi

        export generating_model_base=${logging_bases[j]}
        export generating_experiment_name=${experiment_names_array[j]}

        export data_source="${generating_model_base}/run_0/synthetic_data_circular_statistics_sweep.npy"

        python -m non_parametric_model.scripts.synthetic_data_generation.sweep_circular_statistics \
            --resume_path "${inferring_model_base}/run_0" \
            --synthetic_data_code inference_of_circular_statistics_sweep_from_${generating_experiment_name} \
            --generation_source $data_source

    done

done


