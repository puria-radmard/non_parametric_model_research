# Just a super train on everything!

export yaml_path=$1

export NUMFOLDS=10

for (( fl = 0; fl < $NUMFOLDS; fl++ )); do
    python -m non_parametric_model.scripts.main.training -eap $yaml_path --train_indices_seed $fl
    exit 1
done


# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/mcmaster2022_e1_loccue_est_dim_von_mises.yaml              
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/mcmaster2022_e1_oricue_spike_and_slab_von_mises.yaml
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/mcmaster2022_e1_loccue_est_dim_wrapped_stable.yaml         
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/mcmaster2022_e1_oricue_spike_and_slab_wrapped_stable.yaml
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/bays2009_est_dim_von_mises.yaml                     
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/bays2009_cue_dim_wrapped_stable.yaml                

# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/mcmaster2022_e1_loccue_spike_and_slab_von_mises.yaml       
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/mcmaster2022_e2_dircue_cue_dim_von_mises.yaml
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/bays2009_est_dim_wrapped_stable.yaml                
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/mcmaster2022_e1_loccue_spike_and_slab_wrapped_stable.yaml  
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/mcmaster2022_e2_dircue_cue_dim_wrapped_stable.yaml
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/bays2009_cue_dim_von_mises.yaml                     

# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/mcmaster2022_e2_dircue_est_dim_von_mises.yaml
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/bays2009_spike_and_slab_von_mises.yaml              
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/mcmaster2022_e1_oricue_cue_dim_von_mises.yaml              
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/bays2009_spike_and_slab_wrapped_stable.yaml         
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/mcmaster2022_e1_oricue_cue_dim_wrapped_stable.yaml         
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/mcmaster2022_e1_oricue_est_dim_wrapped_stable.yaml         

# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/mcmaster2022_e2_dircue_est_dim_wrapped_stable.yaml
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/mcmaster2022_e2_dircue_spike_and_slab_von_mises.yaml
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/mcmaster2022_e1_loccue_cue_dim_wrapped_stable.yaml  
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/mcmaster2022_e2_dircue_spike_and_slab_wrapped_stable.yaml
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/mcmaster2022_e1_loccue_cue_dim_von_mises.yaml       
# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/mcmaster2022_e1_oricue_est_dim_von_mises.yaml              


# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/configs/mcmaster2022_e2_dircue_est_dim_von_mises_init_min_seps.yaml

# /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/a_trained_on_data/run_script.sh /homes/pr450/repos/research_projects/error_modelling_torus/non_parametric_model/commands/c_new_data_test/gorgoraptis2011_cue_dim_wrapped_stable.yaml
