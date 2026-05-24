# Project Tree

본 문서는 2026-05 cleanup + Phase A~O + PART6 (head-yaw 정렬 / DEX3 안전화) +
하드웨어 bringup STEP 1~6 후 워크스페이스 구조를 정리한다. 각 파일/디렉토리의
한 줄 역할 + 어느 phase 에서 변경되었는지 참고.

상세 동작은 [`README.md`](README.md) + [`docs/HARDWARE.md`](docs/HARDWARE.md) +
[`docs/INSTALL.md`](docs/INSTALL.md) 참고.

```
G1_Teleoperation/
├── main.py                    Teleoperation entry — SHM owner, worker spawn, CLI (--hand/camera/vr-input/waist/head/lower-body/gait/tactile/no-robot). preflight cleanup + SIGTERM 핸들러 + PIDFILE
├── evaluate.py                External policy eval entry (별도 conda env, --lag-compensate)
├── setup.py                   pip install -e . 진입
├── README.md                  본 워크스페이스 통합 가이드
├── Project_tree.md            (이 파일)
│
├── docs/
│   ├── HARDWARE.md            Quest3/G1/DEX3/Inspire/DXL/ZED/RealSense IDL·msg·Hz·latency (SDK 사실 기반)
│   ├── INSTALL.md             conda env 0→완전한 빌드 절차 (opencv-headless, vuer 0.0.60 + params_proto pin, patch_vuer_xr, logging_mp alias)
│   ├── QUEST3_SETUP.md        Quest3 Linux USB 연결 — adb 설치, udev rule, dev mode 가이드 (jammy 이상 패키지명 반영)
│   ├── DEPLOY_PRO4000.md      Pro4000 배포 + Quest3-only 검증 10-row 체크리스트
│   └── REMAINING_VERIFICATION_TASKS.md  배포 전 잔여 검증 항목 트래커
│
├── scripts/
│   ├── verify_offline.py      하드웨어 0개 검증 — imports, SHM schema, IK build, CLI surface (9 단계)
│   ├── verify_quest3.py       Quest3 + IK 검증 — SHM attach-only reader (2Hz print, --watch)
│   └── patch_vuer_xr.py       vuer 0.0.60 client JS 패치 — hand-tracking 기본값 OFF + WebSocket port 누락 수정 (disable/enable/status)
│
├── sharedmemory/
│   ├── shm_schema.py          모든 SHM layout + HAND/CAMERA/VR_INPUT/WAIST/HEAD mapping. *_ts (int64 ns) Phase D 추가, CAMERA_VIEW role SHM (Part5), 멀티 RealSense (K7)
│   └── shmManager.py          SharedMemoryManager — partial write 지원, schema-based dtype 혼용
│
├── workers/
│   ├── worker_vr.py           Vuer 이벤트 → TELEVISION + QUEST_CONTROLLER SHM. 좌X/Y/우B record 버튼 rising-edge (Phase C)
│   ├── worker_g1_ik.py        SE(3) clutch + HMD R_delta→waist + Right-A 3s cosine recovery + lockout (Phase C). head-yaw 정렬 + rotation/translation 분리 (PART6 §3-A/3-F)
│   ├── worker_g1_ctrl.py      LowCmd_/LowState_ DualRate 50/300Hz, --head off 시 DXL skip (Phase A/F). 29 joint 한정 init (kNotUsedJoint0-5 weight 슬롯 회피)
│   ├── worker_loco.py         (Phase N) --lower-body loco 시 rt/arm_sdk 모드 진입 + LocoClient (thumbstick → Move vx/vy/wz)
│   ├── worker_hand_ctrl.py    Inspire 6 / DEX3 7 motor 양손, trigger toggle, replay/deploy 분기. DEX3 rate-limit + 좌우 거울대칭 (PART6)
│   ├── worker_hand_dds.py     Inspire RH56 Modbus touch sensor poller (DEX3 미해당). DEX3 press_sensor_state length-only logging 지원 (--tactile on, Phase K8)
│   ├── worker_zed.py          ZED stereo direct(USB)/stream + ArUco + depth + workspace mask (Phase F serial 인자)
│   ├── worker_camera.py       RealSense color@30Hz (D435i/455/405). serial 별 인스턴스 → role-based CAMERA SHM (Phase K7 멀티)
│   ├── worker_record.py       FSM 외부 20Hz + RecordCollectors thread + align_and_save_episode (Phase D 78% rewrite). 멀티 카메라 / DEX3 7×2 / modality.json 토글 반영 (K~M)
│   ├── worker_deploy_policy.py  Gr00t_Inference slow(20Hz)/fast(50Hz) + cross-fade + lag-trim (Phase E)
│   ├── worker_plot.py         matplotlib realtime qpos/action plot (50Hz)
│   ├── worker_g1_visualization.py  Meshcat 시각화 (현재 main.py 미사용 — 옵션)
│   ├── keyboard_listener.py   q=shutdown, h=go_home
│   └── A_dual_rate_worker.py  DualRateWorker 기반 클래스 (slow/fast 2-thread)
│
├── g1_control/
│   ├── g1_ik.py               G1_29_ArmIK — Pinocchio reduced robot + CasADi IPOPT, init_pose, smooth_cost
│   ├── g1_whole_control.py    G1_29_ArmController — LowCmd_/LowState_ DDS, leg/waist/arm 분리. damp_to_release(ramp_sec=2.5, kd_hold=5.0) 안전 종료
│   ├── g1_head_dynamixel.py   Dynamixel_Controller — 2모터 syncwrite (Phase A: hardcode override 제거)
│   ├── g1_visualize_whole.py  meshcat 기반 G1 forward visualization
│   ├── remote_controller.py   Unitree 무선 컨트롤러 button KeyMap
│   ├── joint_setting.yaml     kp/kd/default_dof_pos profile (teleop, gr00t, gr00t_zed)
│   └── assets/g1/             URDF + mesh (g1_body29_inspire_zed.urdf 사용)
│       ├── g1_body29_inspire_zed.urdf
│       ├── inspire_hand_left/right.urdf
│       └── meshes/            STL/convex 파일
│
├── hand_control/
│   ├── robot_hand_inspire.py  Inspire_Controller — DDS rt/inspire_hand/{ctrl,state}/{l,r}, mode=0b0001, angle 0..1000
│   ├── robot_hand_dex3.py     Dex3_Controller — DDS rt/dex3/{left,right}/{cmd,state}. _STEP_MAX=0.18, _MAX_POS_ERR=0.30, 좌우 mirror 부호 반전, Index0/Mid0=±90° (공식 spec) (PART6)
│   ├── hand_retargeting.py    HandRetargeting — INSPIRE_HAND / UNITREE_DEX3 (좌/우 enum 순서 다름) yml 기반
│   ├── DEX3-1_spec.md         DEX3-1 공식 가동범위/기구학 spec 요약 (rate-limit 와 cap 산정 근거)
│   ├── inspire_hand/
│   │   ├── inspire_hand.yml
│   │   └── inspire_hand_left/right.urdf
│   ├── unitree_dex3_hand/
│   │   ├── unitree_dex3.yml   (xr_teleoperate 신버전 API 와 호환 위해 단일 target_link_human_indices)
│   │   └── unitree_dex3_left/right.urdf
│   └── dex_retargeting/       구버전 dex_retargeting fork (DexPilot+Vector 분리 전)
│
├── open_television/
│   ├── television.py          Vuer wrapper — HAND_MOVE / CONTROLLER_MOVE / CAMERA_MOVE handler, MotionControllers 분기
│   ├── tv_wrapper.py          OpenXR→Robot 기저 변환 (similarity transform), get_data / get_controller_data
│   └── constants.py           T_robot_openxr, hand2inspire, T_to_unitree_*_wrist 상수
│
├── utils/
│   ├── raw_stream.py          (Phase D 신규) RawStreamBuffer — thread-safe (ts, payload) dedup deque
│   ├── align.py               (Phase D 신규) interp_to_axis (linear/zoh) + common_time_axis (intersection)
│   ├── record_collectors.py   (Phase D 신규) RecordCollectors 3 poller thread + align_and_save_episode. 멀티 카메라 / DEX3 / modality 토글 반영 (K~M)
│   ├── camera_discovery.py    (Phase F 신규) discover_realsense/zed/auto_select, lazy SDK import
│   ├── mat_tool.py            cosine_ease + se3_interp(quat slerp) + fast_mat_inv (Phase C 확장)
│   ├── parquet_sink.py        LeRobot v2.1 parquet writer (Phase D: add_extra_column for raw_ts_*)
│   ├── video_sink.py          mp4 writer (view-dynamic, None-skip)
│   ├── modality_layout.py     (Phase M) modality.json 단일 진실 출처 — hand 종류별 video/state/action 차원 자동 결정
│   ├── frame_utils.py         BGR→RGB + 정합성 검증
│   ├── rate.py                Rate(hz).sleep() / tick_hz()
│   ├── state.py               FSM State enum + next_state + EventsSnapshot
│   ├── weighted_moving_filter.py  IK 후 smooth filter ([0.4,0.3,0.2,0.1], nq)
│   ├── record_config.py       BASE_FOLDER, CHUNK_SIZE 상수
│   ├── cameras.yaml           (Phase F+K7) role↔serial 매핑. 현 운용: ego=D455 046322250265, wrist_l=D405 128422272260, wrist_r=D405 409122271579
│   ├── lan_config.yaml        DDS network_interface (현 데스크톱 enp129s0)
│   ├── act/data2hdf5.py       ACT dataset 변환 보조
│   ├── camera_test/           ZED/RealSense standalone 점검 스크립트
│   └── parquet/
│       ├── build_dataset_meta.py / _v2.py    LeRobot meta jsonl 빌드
│       ├── modality_dex3.template.json       DEX3 7×2 dim 템플릿 (Phase M)
│       └── modality_inspire.template.json    Inspire 6×2 dim 템플릿 (Phase M)
│
├── gui/
│   └── ui_launcher.py         PyQt5 TeleopUI — 카메라뷰 (CAMERA_VIEW role SHM, Part5), 진행률, 터치맵, set/start/reset/replay/deploy/home 버튼. workspace_mask 가드 (검은 화면 회피)
│
├── data_refinement/
│   ├── convert_to_dp.py       LeRobot v2.1 → Diffusion Policy zarr replay buffer
│   ├── convert_to_act.py      LeRobot v2.1 → ACT per-episode HDF5
│   ├── merge_parquet_data.py  여러 task / chunk 합치기
│   ├── sequential_merge.py    episode 순차 병합
│   ├── inspect_parquet.py     parquet schema + row 출력
│   ├── plot_parquet.py        state/action 시계열 시각화
│   ├── apply_mask_to_videos.py  ZED workspace mask post-apply (offline)
│   └── README.md              data_refinement 사용 가이드
│
├── record/                    데이터 저장 root — task별 chunk-XXX/episode_XXXXXX.parquet + videos/
│   └── README.md
│
└── image/                     README 용 다이어그램 (architecture.png, system_conf_exmaple.png)
```

> 2026-05 history rewrite 로 `code_for_kistar_control/` (legacy KISTAR hand) +
> 모든 `.pyc` + `egg-info/` 가 전체 커밋 히스토리에서 제거됨. .git 1.7 GB → 1.6 GB.

## Phase 변경 요약

| Phase | Commit | 핵심 |
|---|---|---|
| 0~6 (cleanup) | ~`5a0053a` | KISTAR/AMO 제거, controller infrastructure, DEX3 양손, RealSense/ZED 양자택일, inspire-only deploy 도입 |
| **A** | `811966d` | DXL hardcode 제거, replay double-counter fix, GUI Deploy → set_start 자동, monocular dead path, deploy 분기 정리 |
| **B** | `c44ee15` | docs/HARDWARE.md 444줄 — SDK 사실 검증 (xr_teleoperate + unitree_sdk2py + televuer) |
| **C** | `95eb95b` | Right-A 3s cosine recovery + lockout, 좌X/Y/우B 컨트롤러 record 버튼, waist anchor=last target |
| **D 1/2** | `d779cf5` | SHM `*_ts` 필드 + writer wiring + raw_stream + align utils |
| **D 2/2** | `a1c31e7` | RecordCollectors + worker_record 78% rewrite + ParquetSink extra_columns |
| **E** | `217bcc9` | Deploy inference-lag chunk trim + `--lag-compensate` CLI |
| **F** | `b3bad1a` | `--waist`/`--head` toggle, 카메라 auto-detect, `--no-robot`, `--zed-mode`, camera_discovery + cameras.yaml |
| **G+H** | `c7fd0e6` | docs/DEPLOY_PRO4000.md, scripts/verify_quest3.py |
| **K (P1-8)** | `1b38d4a` | hand 종류별 modality.json 분기 + 자동 배치 |
| **K7 (P0-3 + P1-4)** | `9db5714` | 멀티 RealSense (ego + wrist_l + wrist_r) + SHM 분리 + view-dynamic VideoSink |
| **K8 (P1-5)** | `516bd88` | 촉각 toggle 골격 (off-경로 100% 불변, length-only 로깅) — `--tactile {off,on}` |
| **K9 (P2)** | `60fb638` | silent fallback 제거 + 중복 dedup 정리 + SLERP TODO |
| **L (Part 2)** | `b15f3be` | eval consistency + IK measurement + Rate phase fix |
| **M (PART3)** | `ab5b1e3` | 모달리티 토글 단일 진실 출처 (modality.json) |
| **N** | `e8c10e3` | `--lower-body {hoist,loco}` + `--gait thumbstick` (motion mode 보행) |
| **Part5** | `54c0ebc` | GUI/Vuer 표시 경로를 CAMERA_VIEW role SHM 로 정합 |
| **O** | `7a963a1` | 새 머신 STEP 1 환경 구축 검증 + INSTALL.md/setup.py 정합 |
| **PART6 ctrl** | `d3f2234` | grip clutch 에 head-yaw 정렬 + rotation/translation 분리 |
| **PART6 dex3** | `c96f66a` | DEX3 rate-limit + 좌우 거울대칭 + 공식 spec 가동범위 |
| **PART6 shutdown** | `6b3a17a` | hoist 종료 시 damp_to_release 로 팔 부드럽게 힘 빼기 |
| **VuerXR fix** | `9cef419`, `2995d94`, `285b0d6` | WebXR hand-tracking 강제 OFF + WebSocket URL port 누락 패치 (`scripts/patch_vuer_xr.py`) |
| **STEP 2~6 bringup** | `6dab495`~`15fc35f` | NIC 갱신, cameras.yaml D405×2, QUEST3 jammy adb, cv2↔PyQt5 Qt 충돌 해결, preflight cleanup + ego mask 가드 |

상세는 [`README.md`](README.md), [`docs/HARDWARE.md`](docs/HARDWARE.md),
[`docs/INSTALL.md`](docs/INSTALL.md).
