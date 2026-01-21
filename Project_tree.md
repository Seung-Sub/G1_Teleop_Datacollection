```
teleop_data_collector
├─ README.md
│
├─ setup.py
├─ act
├─ gr00t
│  ├─ __init__.py
│  ├─ data
│  │  ├─ __init__.py
│  │  ├─ dataset.py
│  │  ├─ embodiment_tags.py
│  │  ├─ schema.py
│  │  └─ transform
│  ├─ eval
│  │  ├─ http_server.py
│  │  ├─ robot.py
│  │  ├─ service.py
│  │  ├─ simulation.py
│  │  └─ wrappers
│  ├─ experiment
│  ├─ model
│  ├─ py.typed
│  ├─ result
│  │  ├─ 0723_apple
│  │  │  └─ apple_checkpoint
│  │  │     ├─ action.zip
│  │  │     └─ checkpoint-100000
│  │  └─ 0801_apple
│  │     └─ apple_checkpoint
│  │        ├─ LoRAx.zip
│  │        └─ checkpoint-100000
│  └─ utils
│
├─ deploy_test_result
|
├─ g1_control
│  ├─ __init__.py
│  ├─ amo
│  │  ├─ adapter_jit.pt
│  │  ├─ adapter_norm_stats.pt
│  │  ├─ amo_jit.pt
│  │  └─ grav_lut_g1.npz
│  ├─ assets
│  │  └─ g1
│  │     ├─ README.md
│  │     ├─ g1_body29_inspire_zed.urdf
│  │     ├─ inspire_hand_left.urdf
│  │     ├─ inspire_hand_right.urdf
│  │     └─ meshes
│  ├─ g1_head_dynamixel.py
│  ├─ g1_ik.py
│  ├─ g1_visualize_whole.py
│  ├─ g1_whole_control.py
│  ├─ grav_lut.py
│  ├─ joint_setting.yaml
│  └─ remote_controller.py
|
├─ record
│  ├─ g1-pick-apple
│  │  ├─ data
│  │  │  └─ chunk-000
│  │  │     ├─ episode_000000.parquet
│  │  │     └─ episode_000149.parquet
│  │  ├─ meta
│  │  │  ├─ episodes.json
│  │  │  ├─ info.json
│  │  │  ├─ modality.json
│  │  │  ├─ stats.json
│  │  │  ├─ tasks.jsonl
│  │  │  └─ tmp
│  │  │     ├─ episodes.jsonl
│  │  │     ├─ info.json
│  │  │     └─ stats.json
│  │  └─ videos
│  │     └─ chunk-000
│  │        └─ observation.images.ego_realsense
│  │           ├─ episode_000000.mp4
│  │           └─ episode_000149.mp4
│  └─ g1-pick-grapes
|
├─ inspire_hand_ws
|
├─ hand_control
│  ├─ __init__.py
│  ├─ dex_retargeting
│  ├─ hand_retargeting.py
│  ├─ inspire_hand
│  │  ├─ inspire_hand.yml
│  │  ├─ inspire_hand_left.urdf
│  │  ├─ inspire_hand_right.urdf
│  │  └─ meshes
│  └─ robot_hand_inspire.py
|
├─ open_television
│  ├─ __init__.py
│  ├─ constants.py
│  ├─ television.py
│  └─ tv_wrapper.py
|
├─ sharedmemory
│  ├─ __init__.py
│  ├─ shmManager.py
│  └─ shm_schema.py
|
├─ gui
│  ├─ __init__.py
│  ├─ teleop_ui.ui
│  └─ ui_launcher.py
|
├─ utils
│  ├─ __init__.py
│  ├─ camera_test
│  │  ├─ camera_test.py
│  │  └─ zed_test.py
│  ├─ frame_utils.py
│  ├─ lan_config.yaml
│  ├─ mat_tool.py
│  ├─ parquet
│  │  ├─ build_dataset_meta.py
│  │  ├─ modality.json
│  │  └─ tasks.jsonl
│  ├─ parquet_sink.py
│  ├─ rate.py
│  ├─ record_config.py
│  ├─ rerun_visualizer.py
│  ├─ state.py
│  ├─ video_sink.py
│  └─ weighted_moving_filter.py
|
├─ workers
│  ├─ __init__.py
│  ├─ A_dual_rate_worker.py
│  ├─ A_worker_example.py
│  ├─ keyboard_listener.py
│  ├─ test_worker_g1.py
│  ├─ test_worker_g1_amo_ik.py
│  ├─ test_worker_hand.py
│  ├─ worker_camera.py
│  ├─ worker_deploy_policy.py
│  ├─ worker_g1_amo.py
│  ├─ worker_g1_ctrl.py
│  ├─ worker_g1_ik.py
│  ├─ worker_g1_visualization.py
│  ├─ worker_hand_ctrl.py
│  ├─ worker_hand_dds.py
│  ├─ worker_plot.py
│  ├─ worker_record.py
│  ├─ worker_vr.py
│  ├─ worker_zed.py
│  └─ worker_zed_slam.py
│
├─ deploy_gr00t.py
└─ main.py

```
