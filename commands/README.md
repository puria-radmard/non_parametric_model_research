Continuing from `.../pre_budapest_commands_and_analysis/commands/post_cajal_model_comparison` on 26/8/24...

After the apparent success of the reparameterisation of the BnP model to:
1. Exclude $\delta = 0 $from the swap function domain
2. Fix $\tilde{\pi}_1$ at 1 but train $\tilde{\pi}_0$
3. Fix inducing points to their gridded locations
I decided to rerun all results with this new model.

Things to catch up on are:
1. 6 model types in total for each of: mcmaster2022_e1_oricue, mcmaster2022_e1_loccue, mcmaster2022_e2_dircue, bays2009, with 10 folds each. Then run the subsequent model comparison/residual kurtosis analysis (`trained_on_data_XXX.yaml`)
2. Model recovery for mcmaster2022_e2_dircue (spike_and_slab_no_uniform_wrapped_stable_medC vs. est_dim_only_no_uniform_wrapped_stable_medC) (`est_dim_vs_sands_model_recovery_XXX.yaml`)
3. Recovery of delobed data for  mcmaster2022_e2_dircue est_dim_only_no_uniform_wrapped_stable_medC (need to redo the way we delobe functions bc now there's no guarantee of total number of lobes!) (`delobed_model_recovery_XXX.yaml`)

Also finally switching to a YAML, and switching out to a better directory naming scheme!



