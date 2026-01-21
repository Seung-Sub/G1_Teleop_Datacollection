# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any
from pathlib import Path

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.model.transforms import GR00TTransform


class BaseDataConfig(ABC):
    @abstractmethod
    def modality_config(self) -> dict[str, ModalityConfig]:
        pass

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


###########################################################################################


class UnitreeG1DataConfig(BaseDataConfig):
    video_keys = ["video.rs_view"]
    state_keys = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)

class UnitreeG1FullBodyDataConfig(UnitreeG1DataConfig):
    video_keys = ["video.rs_view"]
    state_keys = [
        "state.left_leg",
        "state.right_leg",
        "state.waist",
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

class UnitreeG1KistarRealsenseDataConfig(UnitreeG1DataConfig):
    video_keys = ["video.ego_view"]
    state_keys = ["state.waist","state.left_arm","state.right_arm","state.kistar_hand"]
    action_keys = ["action.waist","action.left_arm", "action.right_arm","action.kistar_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

class UnitreeG1KistarZedMonoLeftDataConfig(UnitreeG1DataConfig):
    """ZED 양안 카메라를 지원하는 G1 데이터 설정"""
    video_keys = ["video.ego_left_view"]
    state_keys = ["state.waist","state.left_arm","state.right_arm","state.kistar_hand"]
    action_keys = ["action.waist","action.left_arm", "action.right_arm","action.kistar_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

class UnitreeG1KistarZedMonoRightDataConfig(UnitreeG1DataConfig):
    """ZED 양안 카메라를 지원하는 G1 데이터 설정"""
    video_keys = ["video.ego_right_view"]
    state_keys = ["state.waist","state.left_arm","state.right_arm","state.kistar_hand"]
    action_keys = ["action.waist","action.left_arm", "action.right_arm","action.kistar_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

class UnitreeG1KistarZedDataConfig(UnitreeG1DataConfig):
    """ZED 양안 카메라를 지원하는 G1 데이터 설정"""
    video_keys = ["video.ego_left_view", "video.ego_right_view"]
    state_keys = ["state.waist","state.left_arm","state.right_arm","state.kistar_hand"]
    action_keys = ["action.waist","action.left_arm", "action.right_arm","action.kistar_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

class UnitreeG1KistarInspireKistarDataConfig(UnitreeG1DataConfig):
    """ZED 양안, 왼손 Inspire, 오른손 Kistar 설정"""
    video_keys = ["video.ego_left_view", "video.ego_right_view"]
    state_keys = ["state.waist","state.left_arm","state.right_arm","state.inspire_hand","state.kistar_hand"]
    action_keys = ["action.waist","action.left_arm", "action.right_arm","action.inspire_hand","action.kistar_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

###########################################################################################

DATA_CONFIG_MAP = {
    "unitree_g1": UnitreeG1DataConfig(),
    "unitree_g1_full_body": UnitreeG1FullBodyDataConfig(),
    "unitree_g1_kistar_zed": UnitreeG1KistarZedDataConfig(),
    "unitree_g1_kistar_zed_mono_left": UnitreeG1KistarZedMonoLeftDataConfig(),
    "unitree_g1_kistar_zed_mono_right": UnitreeG1KistarZedMonoRightDataConfig(),
    "unitree_g1_kistar_realsense": UnitreeG1KistarRealsenseDataConfig(),
    "unitree_g1_inspire_kistar": UnitreeG1KistarInspireKistarDataConfig()
}
