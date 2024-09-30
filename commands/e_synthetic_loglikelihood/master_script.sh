# non_parametric_model/commands/e_synthetic_loglikelihood/master_script.sh

# mcmaster2022_e1_oricue_spike_and_slab_wrapped_stable
# mcmaster2022_e2_dircue_spike_and_slab_wrapped_stable
# vandenberg2012_color_spike_and_slab_wrapped_stable

# mcmaster2022_e1_oricue_cue_dim_wrapped_stable
# mcmaster2022_e2_dircue_cue_dim_wrapped_stable
# mcmaster2022_e2_dircue_est_dim_wrapped_stable
# vandenberg2012_color_cue_dim_wrapped_stable


# mcmaster2022_e1_oricue_spike_and_slab_von_mises_and_uniform
# mcmaster2022_e2_dircue_spike_and_slab_von_mises_and_uniform
# vandenberg2012_color_spike_and_slab_von_mises_and_uniform

# mcmaster2022_e1_oricue_cue_dim_von_mises_and_uniform
# mcmaster2022_e2_dircue_cue_dim_von_mises_and_uniform
# mcmaster2022_e2_dircue_est_dim_von_mises_and_uniform
# vandenberg2012_color_cue_dim_von_mises_and_uniform


export real_configs_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/e_synthetic_loglikelihood/configs_real_data"
export synth_configs_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/e_synthetic_loglikelihood/configs_synthetic_data"

# Take just the experiment name
export experiment_name=$1  


# Extract logging base where model from step 1 is saved
export real_data_training_yaml_path="${real_configs_path}/${experiment_name}.yaml"
logging_base=`cat $real_data_training_yaml_path | sed -n -e 's/^.*logging_base: //p' | sed -e 's|["",]||g' | sed -e 's|['',]||g'`
echo
echo logging_base: $logging_base
echo


# Extract how many synthetic sets we are expecting to generate
export synth_data_training_yaml_path="${synth_configs_path}/${experiment_name}.yaml"
num_repeats=`cat $synth_data_training_yaml_path | sed -n -e 's/^.*num_models: //p' | sed -e 's|["",]||g' | sed -e 's|['',]||g'`
echo
echo num_repeats: $num_repeats
echo



# 1. TRAIN MODEL ON REAL DATA
python -m non_parametric_model.scripts.main.training -eap $real_data_training_yaml_path --train_indices_seed 0


# 2. GENERATE SYNTHETIC DATA
python -m non_parametric_model.scripts.synthetic_data_generation.full_distribution --resume_path "${logging_base}/run_0" --num_synthetic_generation_repeats $num_repeats --synthetic_data_code likelihood_dist


# 3. TRAIN ON SYNTHETIC DATA
python -m non_parametric_model.scripts.main.training -eap $synth_data_training_yaml_path --train_indices_seed 0 --synthetic_data_runname run_0

