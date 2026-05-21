# G1_Teleoperation — Part 5: GUI/Vuer 카메라 표시 경로 수정 (CAMERA → CAMERA_VIEW)

> 검증 결과 발견된 **명확한 버그** 수정 지시서. 카메라는 정상 탐색·동작하나, 찾은 프레임이
> GUI/헤드셋 화면까지 도달하지 못한다. 모든 내용은 코드 직접 검증 기반.

---

## 0. 문제 요약 (코드 검증으로 확정)

### 0.1 핵심 버그 — 아무도 안 쓰는 SHM을 읽는다
- **카메라 워커**(worker_camera, worker_zed)는 Phase K7-A 이후 **CAMERA_VIEW 스키마의 role별 SHM**
  (`rs_ego_shm`, `rs_wrist_l_shm`, `rs_wrist_r_shm`)에 `frame_left`/`frame_right`/`frame_ts`/
  `is_stereo` 키로 쓴다. (worker_camera.py:152 `view_shm.write_data(frame_left=...)`)
- **main.py**(64-68행 주석): `camera_shm`(CAMERA 스키마)은 **backward-compat 전용**, 실제 사용은
  role별 CAMERA_VIEW SHM. → **`camera_shm`(CAMERA)에는 아무 워커도 쓰지 않는다.**
- 그러나:
  - **GUI** (`ui_launcher.py` update_frame): `self.camera_shm`(CAMERA)에서 `camera_left`/`realsense`
    키를 읽음. → 항상 0 초기화 프레임 → **검은 화면 / 영상 없음.**
  - **Vuer** (`television.py`:66,251,308): 동일하게 `camera_shm`(CAMERA)에서 `camera_left`/
    `camera_right`/`realsense` 읽음. → 헤드셋에도 영상 없음.

### 0.2 부차 문제 — 멀티 RealSense 표시 UI 부재
- `teleop_ui.ui`: 영상 라벨은 **`zed_video_label` 하나** (1000×700). 뷰 전환은
  `zed_view_toggle_button` 하나로 ZED-left ↔ realsense 2-택. (ui_launcher: `zed_current_view`
  = `ZED_VIEW_LEFT` / `ZED_VIEW_REALSENSE`.)
- 사용자 셋업은 RealSense **3대**(ego D435i + wrist_l/wrist_r D405). → ego/wrist_l/wrist_r 를
  선택해 볼 UI가 없음. ZED 시절 구조에 머묾.

### 0.3 정상인 것 (수정 불필요)
- 카메라 디스커버리(camera_discovery): 제품명 우선순위 매칭, graceful 처리 정상.
- 카메라 워커 → CAMERA_VIEW SHM 쓰기 정상 (수집/학습/eval 경로는 이미 정상 동작).
- GUI 버튼 핸들러(G1/Hand/VR connector, start/home/quit/emergency, record/replay/deploy,
  mask_control): 연결 정상. **버튼 인터랙션은 문제 없음.**
- 경위: Phase K7에서 데이터 경로(워커→SHM→record)를 멀티뷰로 리팩터링할 때 **표시 경로(GUI/Vuer)는
  함께 갱신 안 됨.** 데이터 경로와 표시 경로가 갈라진 것. (책임 아닌 경위.)

---

## 1. [P0] GUI update_frame 을 CAMERA_VIEW role SHM 으로 전환

**파일**: `gui/ui_launcher.py`

### 1.1 SHM attach 변경
현재 (239행): `self.camera_shm = SharedMemoryManager(CAMERA, lock["camera_lock"], names["camera_shm"])`

변경: main.py 가 spawn 한 **활성 카메라 role 들의 CAMERA_VIEW SHM** 을 attach.
```python
from sharedmemory.shm_schema import CAMERA_VIEW
# main.py 가 GUI 에 활성 role 리스트를 전달 (예: ['ego','wrist_l','wrist_r'] 또는 ['ego'] (ZED)).
# ROLE_TO_SHM_KEY = {'ego':'rs_ego_shm','wrist_l':'rs_wrist_l_shm','wrist_r':'rs_wrist_r_shm'}
self.view_shms = {}
for role in self.active_camera_roles:
    shm_key  = ROLE_TO_SHM_KEY[role]
    lock_key = ROLE_TO_LOCK_KEY[role]
    self.view_shms[role] = SharedMemoryManager(CAMERA_VIEW, lock[lock_key], names[shm_key])
```
- `active_camera_roles` 는 §3 에서 main.py 가 전달.
- 기존 `camera_shm`(CAMERA) attach 는 제거하거나, ArUco mask 등 다른 용도로만 쓰면 유지 (확인 필요).

### 1.2 update_frame 읽기 키 변경
현재: `data_dict.get("camera_left")` / `data_dict.get("realsense")`.

변경: 현재 선택된 role 의 CAMERA_VIEW SHM 에서 `frame_left` (+ stereo 면 `frame_right`):
```python
def update_frame(self):
    role = self.current_view_role            # §2 뷰 전환이 설정
    shm = self.view_shms.get(role)
    if shm is None: return
    try:
        d = shm.read_data()
    except Exception:
        return
    is_stereo = int(d.get("is_stereo", 0))
    img = d.get("frame_left", None)          # ZED 좌안 / RealSense mono 공통
    # stereo(ZED) 이고 우안 보기 모드면 frame_right 사용 (선택)
    if img is None:
        # 이전 프레임 재사용 (기존 prev_* 로직 유지)
        img = getattr(self, f"_prev_{role}", None)
        if img is not None: img = img.copy()
    else:
        setattr(self, f"_prev_{role}", img.copy())
    if img is None: return
    # 빈 프레임(전부 0) 가드 — 워커 미동작 시 "신호 없음" 표시
    if not img.any():
        self.zed_video_label.setText(f"[{role}] 영상 신호 없음")
        return
    # ... 기존 QImage 변환/표시 동일 (Format_BGR888) ...
```
- ⚠️ **빈 프레임 가드 추가**: 워커가 아직 프레임을 안 쓴 경우 검은 화면 대신 "신호 없음" 텍스트.
  (현재는 무가드라 카메라 죽어도 검은 화면만 — 디버깅 어려움.)
- ArUco mask 오버레이(APPLY_MASK_IN_GUI)는 ZED-left(ego)일 때만 — 현 로직 유지 가능.

---

## 2. [P0] 뷰 전환을 role 기반으로 (멀티 RealSense 지원)

**파일**: `gui/ui_launcher.py` + `teleop_ui.ui`

### 2.1 최소 변경 (UI 수정 없이 — 우선 권장)
`zed_view_toggle_button` 을 **활성 role 들을 순환**하도록 변경:
```python
# 현재: ZED_VIEW_LEFT <-> ZED_VIEW_REALSENSE 2-택 토글
# 변경: active_camera_roles 를 순환
def on_toggle_view(self):
    roles = self.active_camera_roles            # 예: ['ego','wrist_l','wrist_r']
    i = roles.index(self.current_view_role)
    self.current_view_role = roles[(i + 1) % len(roles)]
    self.zed_video_label.setText("")            # 전환 시 잔상 제거
    # 버튼 라벨도 갱신 (선택): self.zed_view_toggle_button.setText(f"view: {self.current_view_role}")
```
- 단일 라벨 + 토글 버튼만으로 ego→wrist_l→wrist_r→ego 순환. **UI 파일 수정 불필요.**
- `current_view_role` 초기값 = `active_camera_roles[0]` (보통 'ego').

### 2.2 (선택, 후순위) 멀티 패널 동시 표시
3대를 동시에 보려면 `.ui` 에 라벨 2개 추가 (wrist_l/wrist_r 용 작은 라벨). 작업량 크므로 **2.1 로 먼저
충분히 운용 가능**. 동시 표시가 꼭 필요할 때만.

---

## 3. [P0] main.py → GUI 에 활성 role 전달

**파일**: `main.py`, `gui/ui_launcher.py`

- main.py 는 `args.cameras` ([{role,type,serial,name}...]) 로 어떤 카메라를 spawn 하는지 안다.
- GUI 기동(run_ui) 시 **활성 role 리스트**를 넘긴다:
```python
# main.py
active_roles = [c['role'] for c in args.cameras]    # 예: ['ego','wrist_l','wrist_r'] 또는 ['ego']
run_ui(events, shm_names, locks, active_camera_roles=active_roles)
```
- `run_ui` / `TeleopUI.__init__` 시그니처에 `active_camera_roles` 추가. 기본값 `['ego']` (안전).
- 카메라가 ZED 1대면 active_roles=['ego'] → 토글 무동작(순환 길이 1), frame_left 표시. OK.

---

## 4. [P0] Vuer(television.py) 도 동일 전환

**파일**: `open_television/television.py`

> 헤드셋을 목에 걸어 화면을 직접 안 본다 해도, Vuer 는 controller pose 입력 채널로 떠 있어야 하고,
> 화면이 깨지면 WebXR 세션 자체가 불안정할 수 있으니 정합 맞춤.

- 66행 `SharedMemoryManager(CAMERA, ..., "camera_shm")` → ego role 의 CAMERA_VIEW SHM
  (`rs_ego_shm`) attach.
- 251/308행 `data_dict.get("camera_left")` / `get("realsense")` → `get("frame_left")`,
  stereo(ZED)면 `frame_left`/`frame_right` 를 좌/우안으로.
- ⚠️ television.py 의 **HTTPS 자동 fallback 패치**(Phase N 때 제외했던 것)도 이 작업과 함께 커밋.
  카메라 표시 경로 수정과 같은 파일이므로 묶어서 "표시 경로 갱신" 단위로.

---

## 5. [P1] 빈 프레임 / 카메라 미동작 가드

- §1.2 의 빈 프레임 가드를 GUI·Vuer 양쪽에. 워커 미기동/카메라 분리 시 "신호 없음" 표시.
- worker_camera 는 이미 frame miss 시 emergency.set() (확인됨). GUI 가 emergency event 를 감지해
  사용자에게 알리는지 확인 (현재 emergency_button 은 사용자→로봇 방향. 워커→GUI 알림 경로 점검).

---

## 6. 작업 순서

1. **§3** main.py → GUI active_camera_roles 전달 (배선 먼저).
2. **§1** GUI update_frame 을 CAMERA_VIEW role SHM 으로 + 빈 프레임 가드.
3. **§2.1** 뷰 전환 role 순환 (UI 수정 없이).
4. **§4** Vuer 동일 전환 + HTTPS fallback 패치 동봉 커밋.
5. **검증**: RealSense 3대 연결 → GUI 기동 → ego/wrist_l/wrist_r 토글로 각 영상 확인.
   ZED 1대 → frame_left 표시 확인. 카메라 1대 분리 → "신호 없음" 표시 확인.

---

## 7. 검증 체크리스트

- [ ] `--camera realsense` 3대: GUI 에서 toggle 로 ego→wrist_l→wrist_r 순환, 각 실시간 영상 표시.
- [ ] `--camera zed`: frame_left (좌안) 표시. (stereo 우안 보기 옵션 동작 — 선택.)
- [ ] 카메라 1대 분리/미연결: 해당 role 토글 시 "신호 없음" (검은 화면 X, 크래시 X).
- [ ] Vuer 헤드셋: ego 영상 표시 (controller 입력과 무관하게 화면 정합).
- [ ] 녹화/replay/deploy 버튼: 기존대로 동작 (영상 경로 변경이 버튼 로직에 영향 없음 확인).
- [ ] ArUco mask 오버레이(쓰는 경우): ego(ZED-left) 에서 정상.

---

## 8. 원칙

1. 표시 경로(GUI/Vuer)를 데이터 경로(CAMERA_VIEW)와 **단일 정합**. (Part 3 의 단일 출처 철학과 동일.)
2. 빈 프레임 가드로 "조용한 검은 화면" 방지 — 실패가 보이게.
3. UI 파일 수정 최소화 (§2.1 role 순환으로 충분, 멀티 패널은 후순위).
4. 카메라 1대(ZED)~3대(RealSense) 모두 같은 코드 경로 (role 리스트 길이만 다름).
5. HTTPS fallback 등 television.py 변경은 표시 경로 수정과 묶어 한 커밋.
