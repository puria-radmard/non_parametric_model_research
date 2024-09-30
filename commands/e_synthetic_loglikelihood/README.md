Similar to d_finalise_kurtosis_results except we want to do it with likelihood:

1. Train models and get $p(\mathcal{D} | \theta_\mathcal{M}[\mathcal{D}])$, i.e. likelihood of real data under optimised parameters
2. Synthetically generate $K$ datasets $\mathcal{D}[\theta_\mathcal{M}[\mathcal{D}]]$
3. Train models on $K$ synthetic datasets and get $p(\mathcal{D}[\theta_\mathcal{M}[\mathcal{D}]] | \theta_\mathcal{M}[\mathcal{D}[\theta_\mathcal{M}[\mathcal{D}]]])$
4. Repeat for $\mathcal{M}'$ and compare...

