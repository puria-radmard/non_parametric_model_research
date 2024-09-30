# For the est_dim_only and spike_and_slab models of mcmaster2022_e2_dircue_cue_AR2, generate various datasets

# Just a super train on everything!

export yaml_path=$1

export NUMFOLDS=10

for (( fl = 0; fl < $NUMFOLDS; fl++ )); do
    python -m non_parametric_model.scripts.main.training -eap $yaml_path -sdr "run_${fl}" -tis $fl
    exit 1
done


# non_parametric_model/commands/b_lobe_understanding_crossfit/configs/run_script_cross_fit.sh non_parametric_model/commands/b_lobe_understanding_crossfit/configs/cross_fit/est_dim_von_mises_normal.yaml
# non_parametric_model/commands/b_lobe_understanding_crossfit/configs/run_script_cross_fit.sh non_parametric_model/commands/b_lobe_understanding_crossfit/configs/cross_fit/est_dim_von_mises_delobed_flat.yaml
# non_parametric_model/commands/b_lobe_understanding_crossfit/configs/run_script_cross_fit.sh non_parametric_model/commands/b_lobe_understanding_crossfit/configs/cross_fit/est_dim_von_mises_delobed_linear.yaml
# non_parametric_model/commands/b_lobe_understanding_crossfit/configs/run_script_cross_fit.sh non_parametric_model/commands/b_lobe_understanding_crossfit/configs/cross_fit/est_dim_von_mises_delobed_remove.yaml

# non_parametric_model/commands/b_lobe_understanding_crossfit/configs/run_script_cross_fit.sh non_parametric_model/commands/b_lobe_understanding_crossfit/configs/cross_fit/spike_and_slab_von_mises_normal.yaml
# non_parametric_model/commands/b_lobe_understanding_crossfit/configs/run_script_cross_fit.sh non_parametric_model/commands/b_lobe_understanding_crossfit/configs/cross_fit/spike_and_slab_wrapped_stable_normal.yaml

# non_parametric_model/commands/b_lobe_understanding_crossfit/configs/run_script_cross_fit.sh non_parametric_model/commands/b_lobe_understanding_crossfit/configs/cross_fit/est_dim_wrapped_stable_normal.yaml
# non_parametric_model/commands/b_lobe_understanding_crossfit/configs/run_script_cross_fit.sh non_parametric_model/commands/b_lobe_understanding_crossfit/configs/cross_fit/est_dim_wrapped_stable_delobed_flat.yaml
# non_parametric_model/commands/b_lobe_understanding_crossfit/configs/run_script_cross_fit.sh non_parametric_model/commands/b_lobe_understanding_crossfit/configs/cross_fit/est_dim_wrapped_stable_delobed_linear.yaml
# non_parametric_model/commands/b_lobe_understanding_crossfit/configs/run_script_cross_fit.sh non_parametric_model/commands/b_lobe_understanding_crossfit/configs/cross_fit/est_dim_wrapped_stable_delobed_remove.yaml

# non_parametric_model/commands/b_lobe_understanding_crossfit/configs/run_script_cross_fit.sh non_parametric_model/commands/b_lobe_understanding_crossfit/configs/cross_fit/est_dim_von_mises_delobed_flat_init_min_seps.yaml