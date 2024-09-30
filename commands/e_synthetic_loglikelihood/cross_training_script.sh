# non_parametric_model/commands/e_synthetic_loglikelihood/cross_training_script.sh

# TRAINING CODE, GENERATION CODE
# mcmaster2022_e1_oricue_spike_and_slab_von_mises_and_uniform mcmaster2022_e1_oricue_cue_dim_von_mises_and_uniform
# mcmaster2022_e2_dircue_spike_and_slab_von_mises_and_uniform mcmaster2022_e2_dircue_cue_dim_von_mises_and_uniform

export synth_cross_model_configs_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/e_synthetic_loglikelihood/configs_synthetic_different_model"

# Take two experiment names: one for the model being fit, and one for the model from which the data is sourced
export fitting_experiment_name=$1
export generating_experiment_name=$2

export synth_cross_model_data_training_yaml_path="${synth_cross_model_configs_path}/${fitting_experiment_name}_on_${generating_experiment_name}.yaml"
num_repeats=`cat $synth_cross_model_data_training_yaml_path | sed -n -e 's/^.*num_models: //p' | sed -e 's|["",]||g' | sed -e 's|['',]||g'`
echo
echo num_repeats: $num_repeats
echo


python -m non_parametric_model.scripts.main.training -eap $synth_cross_model_data_training_yaml_path --train_indices_seed 0 --synthetic_data_runname run_0

