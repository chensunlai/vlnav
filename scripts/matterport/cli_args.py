# From Isaac Lab's standalone package
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab.envs import ManagerBasedRLEnvCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    arg_group.add_argument("--save_interval", type=int, default=None, help="Interval to save the model.")
    # -- load arguments
    arg_group.add_argument("--resume", type=bool, default=None, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default="", help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def parse_rsl_rl_cfg(task_name, args_cli: argparse.Namespace, play=False) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        rslrl_cfg: The default configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")

    # override the default configuration with CLI arguments
    if args_cli.seed is not None:
        rslrl_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        rslrl_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        rslrl_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        rslrl_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.save_interval is not None:
        rslrl_cfg.save_interval = args_cli.save_interval
    if args_cli.run_name is not None:
        rslrl_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        rslrl_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if rslrl_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        rslrl_cfg.wandb_project = args_cli.log_project_name
        rslrl_cfg.neptune_project = args_cli.log_project_name
    if args_cli.use_cnn is not None:
        rslrl_cfg.use_cnn = args_cli.use_cnn
        rslrl_cfg.policy.class_name = "ActorCriticDepthCNN"
        rslrl_cfg.policy.obs_depth_shape = (24, 32)
    if args_cli.use_rnn:
        rslrl_cfg.policy.rnn_input_size = 2 * rslrl_cfg.policy.actor_hidden_dims[-1]
        rslrl_cfg.policy.rnn_hidden_size = 2 * rslrl_cfg.policy.actor_hidden_dims[-1]
        if not args_cli.use_cnn:
            rslrl_cfg.policy.class_name = "ActorCriticRecurrent"
        else:
            rslrl_cfg.policy.class_name = "ActorCriticDepthCNNRecurrent"
            
    rslrl_cfg.policy.history_length = args_cli.history_length

    return rslrl_cfg
