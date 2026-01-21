# worker_g1_amo.py
import os, sys
base_dir   = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir) 

import time
import threading
import numpy as np

import types
from collections import deque
import torch
import yaml

from workers.A_dual_rate_worker import DualRateWorker  # 기존 DualRateWorker 사용
from utils.rate import Rate

from sharedmemory.shmManager import SharedMemoryManager
from sharedmemory.shm_schema import (
    ROBOT_ACTION, ROBOT_AMO_OBS, ROBOT_AMO_INPUT
)

from g1_control.grav_lut import GravLUT


import logging_mp
logger_mp = logging_mp.get_logger(__name__)

OBS_HZ = 300.0
ACT_HZ = 50.0  # 느린 루프 주파수로 사용

class G1AmoWorker(DualRateWorker):
    """
    빠른 루프(OBS_HZ): RUN 상태에서만 관측 읽고 robot_obs_shm에 씀
    느린 루프(ACT_HZ): RUN 상태에서만 액션 읽고 실제 제어 수행
    상태 전이/초기화도 느린 루프에서 처리
    """
    def __init__(self, shared_event, shm_name, shared_lock):
        super().__init__(slow_hz=ACT_HZ, fast_hz=OBS_HZ)

        # 외부 이벤트
        self._shared_event = shared_event

        # ── SHM ───────────────────────────────────────────
        self.robot_action_shm = SharedMemoryManager(ROBOT_ACTION,       shared_lock["robot_action_lock"], shm_name["robot_action_shm"])
        self.robot_amo_obs_shm    = SharedMemoryManager(ROBOT_AMO_OBS,          shared_lock["robot_amo_obs_lock"],    shm_name["robot_amo_obs_shm"])
        self.robot_amo_input_shm    = SharedMemoryManager(ROBOT_AMO_INPUT,          shared_lock["robot_amo_input_lock"],    shm_name["robot_amo_input_shm"])

        self.device = "cuda"

        self.commands = np.zeros(8, dtype=np.float32)

        with open("g1_control/joint_setting.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        prof = cfg["modes"]["amo"]  # "base" 또는 "amo"

        self.kp  = np.array(prof["kp"], dtype=np.float32)
        self.kd  = np.array(prof["kd"], dtype=np.float32)
        self.default_dof_pos = np.array(prof["amo_default_dof_pos"], dtype=np.float32)
        self.num_actions = int(prof.get("num_actions", 0))
        self.num_dofs = int(prof.get("num_dofs", 0))

        self.arm_dof_lower_range = -0.4 * np.ones(8)
        self.arm_dof_upper_range = 0.4* np.ones(8)

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.action_scale = 0.25
        self.arm_action = self.default_dof_pos[15:]
        self.prev_arm_action = self.default_dof_pos[15:]

        self.scales_ang_vel = 0.25
        self.scales_dof_vel = 0.05

        self.nj = 23
        self.n_priv = 3
        self.n_proprio = 3 + 2 + 2 + 23 * 3 + 2 + 15
        self.history_len = 10
        self.extra_history_len = 25
        self._n_demo_dof = 8

        self.dof_pos = np.zeros(self.nj, dtype=np.float32)
        self.dof_vel = np.zeros(self.nj, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(self.nj)

        self.demo_obs_template = np.zeros((8 + 3 + 3 + 3, ))
        self.demo_obs_template[:self._n_demo_dof] = self.default_dof_pos[15:]
        self.demo_obs_template[self._n_demo_dof+6:self._n_demo_dof+9] = 0.75

        self.target_yaw = 0.0 

        self._in_place_stand_flag = True
        self.gait_cycle = np.array([0.25, 0.25])
        self.gait_freq = 1.3

        self.proprio_history_buf = deque(maxlen=self.history_len)
        self.extra_history_buf = deque(maxlen=self.extra_history_len)
        for i in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
        for i in range(self.extra_history_len):
            self.extra_history_buf.append(np.zeros(self.n_proprio))

        amo_dir = os.path.join(parent_dir, "g1_control/amo")

        # JIT 모델과 통계 파일을 동적으로 로드
        self.policy_jit  = torch.jit.load(os.path.join(amo_dir, "amo_jit.pt"),        map_location=self.device)
        self.adapter     = torch.jit.load(os.path.join(amo_dir, "adapter_jit.pt"),    map_location=self.device)
        norm_stats       = torch.load(     os.path.join(amo_dir, "adapter_norm_stats.pt"))

        self.adapter.eval()
        for param in self.adapter.parameters():
            param.requires_grad = False

        self.input_mean = torch.tensor(norm_stats['input_mean'], device=self.device, dtype=torch.float32)
        self.input_std = torch.tensor(norm_stats['input_std'], device=self.device, dtype=torch.float32)
        self.output_mean = torch.tensor(norm_stats['output_mean'], device=self.device, dtype=torch.float32)
        self.output_std = torch.tensor(norm_stats['output_std'], device=self.device, dtype=torch.float32)

        self.adapter_input = torch.zeros((1, 8 + 4), device=self.device, dtype=torch.float32)
        self.adapter_output = torch.zeros((1, 15), device=self.device, dtype=torch.float32)

        lut_path = os.path.join(amo_dir, "grav_lut_g1.npz")  # 경로는 환경에 맞게
        self.gravlut = GravLUT(lut_path)


    def quatToEuler(self,quat):
        eulerVec = np.zeros(3)
        qw = quat[0] 
        qx = quat[1] 
        qy = quat[2]
        qz = quat[3]
        # roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if np.abs(sinp) >= 1:
            eulerVec[1] = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            eulerVec[1] = np.arcsin(sinp)

        # yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)
        
        return eulerVec

    def get_observation(self, pelvis_height, torso_quat, vel_command):
        rpy = self.quatToEuler(self.quat)
        torso_rpy = self.quatToEuler(torso_quat)

        #command 0 : vx 앞뒤
        #command 1 : yaw 로봇이 보는 방향?
        #command 2 : vy 좌우

        v_x = np.clip(vel_command[0], -1.0, 1.0)   # vx
        v_y = np.clip(vel_command[1], -1.0, 1.0)   # vy
        # vel_command[2] = np.clip(vel_command[2], -1.0, 1.0)   # yaw
        pelvis_height = np.clip(pelvis_height, -0.25, 0.0) # height

        self.target_yaw = vel_command[2] #velocidty yaw #좌우


        # dyaw = rpy[2] - self.target_yaw
        # dyaw = np.remainder(dyaw + np.pi, 2 * np.pi) - np.pi
        # if self._in_place_stand_flag:
        #     dyaw = 0.0

        # 보행 제어 부분 
        SPEED_THRESH = 0.1     # m/s      : 걷기‧뛰기 전환 임계값
        YAW_THRESH   = 0.05    # rad ≈ 3° : 회전 시작/정지 임계값

        # ① 평면 속도 크기
        speed = np.hypot(v_x, v_y)

        # ② 현재-목표 yaw 차이(–π~π)
        dyaw = rpy[2] - self.target_yaw
        dyaw = np.remainder(dyaw + np.pi, 2 * np.pi) - np.pi
        dyaw_abs = np.abs(dyaw)


        obs_dof_vel = self.dof_vel.copy()
        obs_dof_vel[[4, 5, 10, 11, 13, 14]] = 0.0

        gait_obs = np.sin(self.gait_cycle * 2 * np.pi)

        self.adapter_input = np.concatenate([np.zeros(4), self.dof_pos[15:]])

        self.adapter_input[0] = 0.75 + pelvis_height    # base height
        self.adapter_input[1] = torso_rpy[2]           # torso yaw
        self.adapter_input[2] = torso_rpy[1]           # torso pitch
        self.adapter_input[3] = torso_rpy[0]           # torso roll

        self.adapter_input = torch.tensor(self.adapter_input).to(self.device, dtype=torch.float32).unsqueeze(0)
            
        self.adapter_input = (self.adapter_input - self.input_mean) / (self.input_std + 1e-8)
        self.adapter_output = self.adapter(self.adapter_input.view(1, -1))
        self.adapter_output = self.adapter_output * self.output_std + self.output_mean

        obs_prop = np.concatenate([
                    self.ang_vel * self.scales_ang_vel,
                    rpy[:2],
                    (np.sin(dyaw),
                    np.cos(dyaw)),
                    (self.dof_pos - self.default_dof_pos),
                    self.dof_vel * self.scales_dof_vel,
                    self.last_action,
                    gait_obs,
                    self.adapter_output.cpu().numpy().squeeze(),
        ])

        obs_priv = np.zeros((self.n_priv, ))
        obs_hist = np.array(self.proprio_history_buf).flatten()

        obs_demo = self.demo_obs_template.copy()
        obs_demo[:self._n_demo_dof] = self.dof_pos[15:]
        
        obs_demo[self._n_demo_dof] = v_x   #앞뒤
        obs_demo[self._n_demo_dof+1] = v_y


        # self._in_place_stand_flag = np.abs(v_x) < 0.1
        
        self._in_place_stand_flag = (speed <= SPEED_THRESH) and (dyaw_abs <= YAW_THRESH)


        obs_demo[self._n_demo_dof+3] = torso_rpy[2]
        obs_demo[self._n_demo_dof+4] = torso_rpy[1]
        obs_demo[self._n_demo_dof+5] = torso_rpy[0]
        obs_demo[self._n_demo_dof+6:self._n_demo_dof+9] = 0.75 + pelvis_height

        self.proprio_history_buf.append(obs_prop)
        self.extra_history_buf.append(obs_prop)
        
        return np.concatenate((obs_prop, obs_demo, obs_priv, obs_hist))

    # 느린 루프: 상태 전이 + (RUN일 때) 액션→제어 입력
    def do_slow(self) -> None:
        # 입력 읽기 
        amo_intput = self.robot_amo_input_shm.read_data()
        obs = self.get_observation(amo_intput["pelvis_height"], amo_intput["torso_quat"], amo_intput["vel_command"])
        
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            extra_hist = torch.tensor(np.array(self.extra_history_buf).flatten().copy(), dtype=torch.float).view(1, -1).to(self.device)
            raw_action = self.policy_jit(obs_tensor, extra_hist).cpu().numpy().squeeze()
        
        raw_action = np.clip(raw_action, -40., 40.)
        self.last_action = np.concatenate([raw_action.copy(), (self.dof_pos - self.default_dof_pos)[15:] / self.action_scale])
        scaled_actions = raw_action * self.action_scale

        pd_target = np.concatenate([scaled_actions, np.zeros(8)]) + self.default_dof_pos

        self.gait_cycle = np.remainder(self.gait_cycle + (1.0/ACT_HZ) * self.gait_freq, 1.0)
        if self._in_place_stand_flag and ((np.abs(self.gait_cycle[0] - 0.25) < 0.05) or (np.abs(self.gait_cycle[1] - 0.25) < 0.05)):
            self.gait_cycle = np.array([0.25, 0.25])
        if (not self._in_place_stand_flag) and ((np.abs(self.gait_cycle[0] - 0.25) < 0.05) and (np.abs(self.gait_cycle[1] - 0.25) < 0.05)):
            self.gait_cycle = np.array([0.25, 0.75])
        # 3) 팔 제어 명령 전송

        leg_tau_ff, waist_tau_ff = self.gravlut.ff_from_q(self.dof_pos)


        self.robot_action_shm.write_data(
            action_leg = pd_target[:12],
            action_leg_tauff = leg_tau_ff,
            action_waist = pd_target[12:15],
            action_waist_tauff = waist_tau_ff
        )

    # 빠른 루프: RUN에서만 관측(250 Hz) → SHM 기록 (+옵션: 주파수 측정)
    def do_fast(self) -> None:


        amo_obs = self.robot_amo_obs_shm.read_data()

        self.dof_pos = amo_obs["amo_q"]
        self.dof_vel = amo_obs["amo_dq"]
        self.quat = amo_obs["quat"]
        self.ang_vel = amo_obs["ang_vel"]
        self.ang_vel = np.asarray(self.ang_vel).flatten()


    # 리소스 정리
    def stop(self) -> None:
        super().stop()
        try:
            self.robot_action_shm.worker_close()
            self.robot_amo_obs_shm.worker_close()
            self.robot_amo_input_shm.worker_close()
        finally:
            logger_mp.info("[G1_Ctrl] 종료 및 SHM 정리 완료")


# ────────────── 실행 진입점 예시 ──────────────
def worker_g1_amo(shared_event, shm_name, shared_lock):
    w = G1AmoWorker(shared_event, shm_name, shared_lock)
    try:
        w.start()
        while not shared_event['shutdown'].is_set():
            time.sleep(0.1)
    finally:
        w.stop()
