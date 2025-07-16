# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/Vlnav-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
gym.register(
    id="Vlnav-Velocity-Matterport-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.matterport_env_cfg:UnitreeGo2MatterportEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2MatterportPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_matterport_ppo_cfg.yaml",
    },
)