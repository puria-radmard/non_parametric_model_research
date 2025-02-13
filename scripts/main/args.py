import os

from purias_utils.error_modelling_torus.data_utils.loading_utils import dataset_choices
from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model import VALID_EMISSION_TYPES
from purias_utils.util.arguments_yaml import ConfigNamepace

import argparse

#############################################

yaml_path_parser = argparse.ArgumentParser()
yaml_path_parser.add_argument('--experiment_arguments_path', '-eap', required=True, type=str, nargs='+')
yaml_path_parser.add_argument('--train_indices_seed', '-tis', required=False, type=int)
yaml_path_parser.add_argument('--train_indices_path', '-tip', required=False, type=str)
yaml_path_parser.add_argument('--resume_path', '-rp', required=False, type=str)
args_for_args, _ = yaml_path_parser.parse_known_args()



new_args = ConfigNamepace.from_yaml_path(args_for_args.experiment_arguments_path[0])

for eap in args_for_args.experiment_arguments_path[1:]:
    added_args = ConfigNamepace.from_yaml_path(eap)
    new_args.update(added_args)

if 'logging_base' not in new_args.dict.keys():
    yaml_path_parser.add_argument('--logging_base', '-lb', required=True, type=str)


if new_args.dataset_name == 'cross_model_fit':
    yaml_path_parser.add_argument('--synthetic_data_runname', '-sdr', required=True, type=str)


args_for_args, _ = yaml_path_parser.parse_known_args()


args = ConfigNamepace({})
for defaults_path in new_args.dict.get('defaults_paths', []):                   # only does one layer at the moment!
    args.update(ConfigNamepace.from_yaml_path(defaults_path))

args.update(new_args)
args.update(ConfigNamepace(vars(args_for_args)))


#############################################

try:
    beta = float(args.beta)
    # assert beta >= 0.0
except:
    beta = args.beta
    assert beta == 'nat', beta

dataset_name = args.dataset_name

M_batch = args.M_batch
M_test_per_set_size = args.M_test_per_set_size

trainable_kernel_delta = False
fix_non_swap = args.fix_non_swap
include_pi_1_tilde = args.include_pi_1_tilde
if include_pi_1_tilde:
    assert fix_non_swap
fix_inducing_point_locations = args.fix_inducing_point_locations
remove_uniform = args.remove_uniform
include_pi_u_tilde = args.include_pi_u_tilde
emission_type = args.emission_type
swap_type = args.swap_type

num_models = args.num_models


#############
# Might have to infer resume_path, only case I can think of is for data generation...
if args.resume_path is not None:
    if not os.path.exists(args.resume_path):
        actual_resume_path = os.path.join(args.logging_base, args.resume_path)
        args.update(ConfigNamepace({"resume_path": actual_resume_path}))
        assert os.path.exists(args.resume_path)
#############

#############
# Allow easy iteration
if args.dataset_name == 'cross_model_fit':
    assert args.synthetic_data_runname is not None
    assert args.synthetic_data_code is not None

    actual_synthetic_data_path = os.path.join(args.synthetic_data_root, args.synthetic_data_runname)
    args.update(ConfigNamepace({"synthetic_data_path": actual_synthetic_data_path}))
    assert os.path.exists(args.synthetic_data_path), args.synthetic_data_path
#############


logging_frequency = args.logging_frequency if args.logging_frequency is not None else 100
testing_frequency = args.testing_frequency if args.testing_frequency is not None else 100

assert not ((args.train_indices_path is not None) and (args.train_indices_seed is not None))


assert args.dataset_name in dataset_choices + ['misattribution_analysis', 'cross_model_fit'], dataset_choices + ['misattribution_analysis', 'cross_model_fit']
assert args.emission_type in VALID_EMISSION_TYPES + ['residual_deltas']
assert args.swap_type in ['full', 'fixed_cue', 'cue_dim_only', 'est_dim_only', 'spike_and_slab', 'no_dim']


if args.emission_type == 'residual_deltas':
    assert args.M_batch <= 0, "emission_type=residual_deltas only implemented for full dataset batching right now! (--M_batch 0)"
    assert args.remove_uniform, "emission_type=residual_deltas only implemented for no uniform component right now! (--remove_uniform)"
