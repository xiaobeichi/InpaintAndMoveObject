import argparse
import torch
from os import path


def get_config():
    config = argparse.ArgumentParser()

    # basic hyperparams to specify where to load/save data from/to
    # config.add_argument("--log_dir", type=str, default="log")
    config.add_argument("--dataset_name", type=str, default="llff")
    config.add_argument("--data_dir", type=str, default="data")
    # model hyperparams
    config.add_argument("--rawnerf_mode", action="store_true")
    config.add_argument("--use_viewdirs", action="store_false")
    config.add_argument("--randomized", action="store_false")
    config.add_argument("--ray_shape", type=str, default="cone")  # should be "cylinder" if llff
    config.add_argument("--white_bkgd", action="store_false")  # should be False if using llff
    config.add_argument("--override_defaults", action="store_true")
    config.add_argument("--num_levels", type=int, default=2)
    config.add_argument("--num_samples", type=int, default=128)
    config.add_argument("--hidden", type=int, default=256)
    config.add_argument("--density_noise", type=float, default=0.0)
    config.add_argument("--density_bias", type=float, default=-1.0)
    config.add_argument("--rgb_padding", type=float, default=0.001)
    config.add_argument("--resample_padding", type=float, default=0.01)
    config.add_argument("--min_deg", type=int, default=0)
    config.add_argument("--max_deg", type=int, default=16)
    config.add_argument("--viewdirs_min_deg", type=int, default=0)
    config.add_argument("--viewdirs_max_deg", type=int, default=4)
    # loss and optimizer hyperparams
    config.add_argument("--coarse_weight_decay", type=float, default=0.1)
    config.add_argument("--lr_init", type=float, default=1e-3)
    config.add_argument("--lr_final", type=float, default=5e-5)
    config.add_argument("--lr_delay_steps", type=int, default=2500)
    config.add_argument("--lr_delay_mult", type=float, default=0.1)
    config.add_argument("--weight_decay", type=float, default=1e-5)
    config.add_argument("--reg_weight", type=float, default=1e-5)
    # training hyperparams
    config.add_argument("--factor", type=int, default=2)
    config.add_argument("--max_steps", type=int, default=200_000)
    config.add_argument("--batch_size", type=int, default=2048)
    config.add_argument("--do_eval", action="store_false")
    config.add_argument("--continue_training", action="store_true")
    config.add_argument("--save_every", type=int, default=1000)
    config.add_argument("--test_image", type=int, default=5000)
    config.add_argument("--device", type=str, default="cuda")
    # visualization hyperparams
    config.add_argument("--chunks", type=int, default=8192)
    config.add_argument("--model_weight_path", default="log/model.pt")
    config.add_argument("--visualize_depth", action="store_true")
    config.add_argument("--visualize_normals", action="store_true")
    # extracting mesh hyperparams
    config.add_argument("--x_range", nargs="+", type=float, default=[-1.2, 1.2])
    config.add_argument("--y_range", nargs="+", type=float, default=[-1.2, 1.2])
    config.add_argument("--z_range", nargs="+", type=float, default=[-1.2, 1.2])
    config.add_argument("--grid_size", type=int, default=256)
    config.add_argument("--sigma_threshold", type=float, default=50.0)
    config.add_argument("--occ_threshold", type=float, default=0.2)

    config = config.parse_args()
    config.save_dir = path.join(config.data_dir, "checkpoints")
    config.model_weight_path = path.join(config.data_dir, "checkpoints/model.pt")

    # default configs for llff, automatically set if dataset is llff and not override_defaults
    if config.dataset_name == "llff" and not config.override_defaults:
        config.factor = 4 #!!
        config.ray_shape = "cylinder"
        config.white_bkgd = False
        config.density_noise = 1.0


    return config
