from .dex_retargeting.retargeting_config import RetargetingConfig
from pathlib import Path
import yaml
from enum import Enum
import os
import logging_mp
logger_mp = logging_mp.get_logger(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)

# URDF 기본 탐색 디렉터리를 hand_control로 설정
RetargetingConfig.set_default_urdf_dir(os.path.join(parent_dir, "hand_control"))



class HandType(Enum):
    INSPIRE_HAND = "hand_control/inspire_hand/inspire_hand.yml"
    INSPIRE_HAND_Unit_Test = "hand_control/inspire_hand/inspire_hand.yml"

class HandRetargeting:
    def __init__(self, hand_type: HandType):
        # 필요 시 타입별로 URDF 디렉터리 재설정 (현재는 공통 hand_control 사용)
        hand_control_dir = os.path.join(parent_dir, "hand_control")
        RetargetingConfig.set_default_urdf_dir(hand_control_dir)

        config_file_path = Path(hand_type.value)

        try:
            with config_file_path.open('r') as f:
                self.cfg = yaml.safe_load(f)

            # 좌/우 어느 한쪽만 있어도 동작하도록 유연화
            self.left_retargeting = None
            self.right_retargeting = None
            self.left_retargeting_joint_names = None
            self.right_retargeting_joint_names = None

            if 'left' in self.cfg:
                left_retargeting_config = RetargetingConfig.from_dict(self.cfg['left'])
                self.left_retargeting = left_retargeting_config.build()
                self.left_retargeting_joint_names = self.left_retargeting.joint_names

            if 'right' in self.cfg:
                right_retargeting_config = RetargetingConfig.from_dict(self.cfg['right'])
                self.right_retargeting = right_retargeting_config.build()
                self.right_retargeting_joint_names = self.right_retargeting.joint_names

            if self.left_retargeting is None and self.right_retargeting is None:
                raise ValueError("Configuration file must contain at least one of 'left' or 'right' keys.")



            # Inspire 전용 하드웨어 매핑 (기존 로직 유지)
            if hand_type in (HandType.INSPIRE_HAND, HandType.INSPIRE_HAND_Unit_Test):
                self.left_inspire_api_joint_names  = [
                    'L_pinky_proximal_joint', 'L_ring_proximal_joint', 'L_middle_proximal_joint',
                    'L_index_proximal_joint', 'L_thumb_proximal_pitch_joint', 'L_thumb_proximal_yaw_joint'
                ]
                self.right_inspire_api_joint_names = [
                    'R_pinky_proximal_joint', 'R_ring_proximal_joint', 'R_middle_proximal_joint',
                    'R_index_proximal_joint', 'R_thumb_proximal_pitch_joint', 'R_thumb_proximal_yaw_joint'
                ]
                self.left_dex_retargeting_to_hardware = (
                    [self.left_retargeting_joint_names.index(name) for name in self.left_inspire_api_joint_names]
                    if self.left_retargeting else None
                )
                self.right_dex_retargeting_to_hardware = (
                    [self.right_retargeting_joint_names.index(name) for name in self.right_inspire_api_joint_names]
                    if self.right_retargeting else None
                )


        except FileNotFoundError:
            logger_mp.error(f"Configuration file not found: {config_file_path}")
            raise
        except yaml.YAMLError as e:
            logger_mp.error(f"YAML error while reading {config_file_path}: {e}")
            raise
        except Exception as e:
            logger_mp.error(f"An error occurred: {e}")
            raise