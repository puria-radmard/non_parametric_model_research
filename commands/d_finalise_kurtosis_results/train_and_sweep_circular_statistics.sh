export configs_path="/homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/d_finalise_kurtosis_results/configs"

export experiment_name=$1   # e.g. mcmaster2022_e1_oricue_cue_dim_wrapped_stable, mcmaster2022_e1_oricue_spike_and_slab_wrapped_stable mcmaster2022_e1_oricue_cue_dim_von_mises_and_uniform


export training_yaml_path="${configs_path}/${experiment_name}.yaml"
logging_base=`cat $training_yaml_path | sed -n -e 's/^.*logging_base: //p' | sed -e 's|["",]||g' | sed -e 's|['',]||g'`
echo $logging_base

# 1. train model
# XXX: train full model!
python -m non_parametric_model.scripts.main.training -eap $training_yaml_path --train_indices_seed 0

# 2. generate (sweep) synthetic data
python -m non_parametric_model.scripts.synthetic_data_generation.sweep_circular_statistics --resume_path "${logging_base}/run_0"

# 3. generate (sweep) synthetic data with forced spike_and_slab
python -m non_parametric_model.scripts.synthetic_data_generation.sweep_circular_statistics --resume_path "${logging_base}/run_0" --synthetic_data_code circular_statistics_sweep_spike_and_slab --make_spike_and_slab

# 4. Do cross inference sweep (between BnP and forced spike_and_slab)   --- see also: sweep_circular_statistics_across_trained_models.sh
export normal_data_path="${logging_base}/run_0/synthetic_data_circular_statistics_sweep.npy"
export sands_data_path="${logging_base}/run_0/synthetic_data_circular_statistics_sweep_spike_and_slab.npy"
python -m non_parametric_model.scripts.synthetic_data_generation.sweep_circular_statistics --resume_path "${logging_base}/run_0" --synthetic_data_code inference_of_circular_statistics_sweep_with_spike_and_slab --generation_source $sands_data_path
python -m non_parametric_model.scripts.synthetic_data_generation.sweep_circular_statistics --resume_path "${logging_base}/run_0" --synthetic_data_code inference_of_circular_statistics_sweep_spike_and_slab_with_bnp --make_spike_and_slab --generation_source $sands_data_path

# 5. TODO: train a spike and slab + wS model on this BnP + von Mises 
