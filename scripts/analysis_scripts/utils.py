import numpy as np

# Evaluate calibration curve for this repititon (see latex doc for notation used!)
def approximate_cdf(cdf_eval, cdf_eval_loc, eval_loc):
    "Interpolate CDF on circle"
    # print('bring these checks back and make sure they work!!!')
    assert np.isclose(cdf_eval_loc[0], -np.pi)# and np.isclose(cdf_eval[0], 0.0)
    assert np.isclose(cdf_eval_loc[-1], +np.pi)# and np.isclose(cdf_eval[-1], 1.0)
    assert (eval_loc.min() > -np.pi) and (eval_loc.max() < +np.pi)
    assert len(cdf_eval.shape) == len(cdf_eval_loc.shape) == len(eval_loc.shape) == 1
    
    # Reshape to make comparison compact
    num_axes = len(eval_loc.shape)
    cdf_eval_loc_reshaped = cdf_eval_loc.reshape(1, -1)
    eval_loc_reshaped = eval_loc.reshape(-1, 1)

    # Interpolation bounds
    lower_idx = (eval_loc_reshaped >= cdf_eval_loc_reshaped).sum(-1) - 1
    assert lower_idx.min() >= 0
    lower_eval = cdf_eval[lower_idx]
    upper_eval = cdf_eval[lower_idx+1]

    # Interpolation
    lower_ratio = (eval_loc - cdf_eval_loc[lower_idx]) / (cdf_eval_loc[lower_idx+1] - cdf_eval_loc[lower_idx])
    upper_ratio = 1. - lower_ratio
    assert lower_ratio.min() >= 0.0 and lower_ratio.max() <= 1.0
    return (lower_ratio * lower_eval) + (upper_ratio * upper_eval)


