# Inspire RH56E2 (FTP) — DEX3-parity Teleop/Record/Deploy 통합 설계

작성일: 2026-06-06
대상: `/home/kist/G1_Teleop_Datacollection`
목표: 기존 DEX3-1 기반 Quest3 Teleop · Data collection · Policy deploy 전체 파이프라인을,
Inspire RH56E2(=RH56DFTP, Unitree 부품명 RH56DFQ-2L/2R) 5지 핸드로도 **이질감 없이** 동작하게 한다.

근거 스펙: `hand_control/Inspire_RH56E2_FTP_detailed_spec.md`.
모든 결정은 실제 소스(아래 "검증된 사실") 확인 후 작성. 추측 배제.

---

## 1. 검증된 사실 (구현 전제)

### 1.1 통신 토폴로지
- Inspire 핸드는 **Modbus-TCP** 서버: 오른손 `192.168.123.210:6000`, 왼손 `192.168.123.211:6000`.
- `workers/worker_hand_dds.py` 가 `inspire_sdkpy.inspire_sdk.ModbusDataHandler` 를 띄워
  **DDS ↔ Modbus 브리지** 역할: `rt/inspire_hand/ctrl/{l,r}` 구독→Modbus write,
  Modbus read→`rt/inspire_hand/state/{l,r}` · `touch/{l,r}` publish.
- 따라서 `Inspire_Controller` 는 **DDS 만** 사용 (ctrl publish + state subscribe).

### 1.2 DDS 메시지 필드 (inspire_sdkpy 실소스 확인)
- `inspire_hand_ctrl`: `pos_set[6] angle_set[6] force_set[6] speed_set[6]`(int16), `mode`(int8).
- `inspire_hand_state`: `pos_act[6] angle_act[6] force_act[6] current[6]`(int16),
  `err[6] status[6] temperature[6]`(uint8). (필드명은 `err`, `temperature`.)
- `mode` 는 **비트마스크** (`inspire_sdk.py:116-130`):
  `0b0001`=angle, `0b0010`=pos, `0b0100`=force, `0b1000`=speed.
  → `mode=0b1101` 한 메시지로 angle+force+speed 동시 전송 가능.
- 단위/범위(스펙 §4): `angle_set/angle_act` 0–1000 (1000=open, 0=closed),
  `force_set` 0–1000(g), `speed_set` 0–1000(1000≈800ms full travel).
- DDS ctrl 로는 `CURRENT_LIMIT 설정 / CLEAR_ERROR / SAVE` **불가** (Modbus 전용).

### 1.3 기존 파이프라인 (이미 Inspire 대응)
- `worker_hand_ctrl.py`: `hand=="inspire"` 분기로 record/replay/deploy 12 DOF 이미 처리.
  SHM 은 14D 패딩, Inspire 는 `[:12]` 사용.
- record(`utils/record_collectors.py`, `utils/modality_layout.build_state_layout('inspire')`):
  6+6=12, state/action idx `[19:25]/[25:31]` 정확. **변경 불필요.**
- 저장 단위: state=`angle_act/1000`(0..1), action=목표 q(0..1). 둘 다 0..1 정규화 → 일관.

### 1.4 발견된 버그 (Inspire 동작 직결)
- `worker_deploy_dp.py:201`: 오른손 슬라이스가 `14+7`(DEX3 7+7 가정) 고정.
  Inspire 학습 action 은 6+6 연속(오른손 col 14+hd)이라 mis-slice → DP 정책 배포 시 손가락 어긋남.
  (`convert_to_dp.py` 는 parquet action 컬럼을 그대로 복사하므로 layout = record layout = 6+6 연속.)
  GR00T 경로(`worker_deploy_policy.py`)는 modality 이름 기반이라 정상.
- `worker_hand_ctrl.py:342`: deploy inspire clip 이 `a_max=1.0` 만 → 음수 q 미방어.
- `robot_hand_inspire.py`: 구식 폴링 subscribe + `any(right_hand_state_array)` 단일 대기(왼손 미확인),
  ctypes Array truthiness(`if dual_hand_state_array and ...`), 오기 주석("milli-radian→radian").

---

## 2. 설계 결정 (사용자 승인됨)

1. **안전 제어 = DDS force/speed + telemetry 모니터** (Modbus 사이드채널 없음).
2. **손가락 모드 인터페이스 = CLI 인자**.
3. **데이터 저장 = DOF별 정규화 각도 6+6** (현행 유지).
4. **패리티에 필요한 버그는 함께 수정**.

추가 가이드(사용자): "어떤 물체든 firm 하게 파지·유지, 속도 빠르게, 빈손에서도 끝까지,
모터 과부하로 꺼지지 않게." → Inspire 는 force feedback 폐루프라
`speed_set=full`(빠름) + `force_set` 상한(firm, 도달 시 STATUS=3 정지) + 펌웨어 current 보호가
정확히 이 요구를 만족. DEX3 의 rate-limit(위치제어 워크어라운드)은 **복제하지 않음**.

---

## 3. 구현 항목

### 3.1 `hand_control/robot_hand_inspire.py` (핵심 재작성)
- **DDS init 견고화**: 폴링 `_subscribe_hand_state` → 콜백 `.Init(handler, 1)`.
  양손 `recv_ts>0` 까지 대기. (DEX3 패턴 이식.)
- **telemetry 수신**: 콜백에서 `angle_act`(state) + `force_act/current/err/status/temperature`
  를 좌/우 배열에 저장. fault 코드 변경 시 `get_error_description()` 로 디코딩 후 rate-limited 로깅.
- **`ctrl_dual_hand` 안전 제어**: 매 publish
  - `angle_set = clamp(int(q*1000), 0, 1000)`,
  - `force_set = [grip_force]*6`, `speed_set = [grip_speed]*6` (init 시 1회 세팅),
  - `mode = 0b1101`,
  - **per-DOF safe-hold**: `status∈{5,6,7}` 또는 `err≠0` 또는 `temperature≥75℃` 인 DOF 는
    이번 사이클 목표를 현재 `angle_act` 로 대체(추가 가압 중단). 다음 사이클 telemetry 로 재평가
    → fault 해소 시 자동 재개. firmware 가 실제 cutoff 방지, 이 hold 는 faulted DOF 반복 가압 억제.
  - teleop·replay·deploy 모두 `ctrl_dual_hand` 경유 → 안전 제어 자동 적용.
- **controller-mode grip 벡터 (CLI 설정 기반)**: DEX3 패턴과 동일하게
  - idx 0..3 = pinky/ring/middle/index, idx4=thumb_bend, idx5=thumb_yaw (open=1.0, closed=0.0).
  - 엄지(idx4,5)는 **항상** `thumb_bend/thumb_yaw` 자세 (DEX3 의 thumb proximal 고정과 동일 의미).
  - grasp 토글 시 `grasp_fingers` 에 포함된 idx0..3 만 `closed_q=max(0,1-close_depth)` 로 닫힘,
    미포함/release 는 open(1.0).
  - rising-edge 토글·home 시 release+lockout 동작은 현행 유지.
- **데이터/replay/deploy**: state=`angle_act/1000`(0..1), action=목표 q(0..1) — 절대값이라
  모드 무관 replay/deploy 직송. (주석 정정 포함.)
- hand-mode(retargeting) 로직·tactile 은 범위 밖(현행 유지).

### 3.2 CLI (`main.py`)
신규 인자 (Inspire controller-mode 전용, 기존 `--thumb-bend/--thumb-yaw` 패턴 확장):
- `--grasp-fingers` (default `pinky,ring,middle,index`): 파지 시 닫히는 손가락 subset(comma).
  엄지는 항상 자세 지정이므로 미포함. 예: `--grasp-fingers index,middle`.
- `--close-depth` (float, default 1.0): 파지 깊이 0..1 (1.0=완전 폐쇄).
- `--grip-force` (int, default 800, 0..1000): force_set. firm 파지+열 마진. 최대 1000.
- `--grip-speed` (int, default 1000, 0..1000): speed_set. 기본 full.
인자 흐름: `main.py`(inspire spec args 튜플) → `worker_hand_ctrl`(신규 kwargs, default 보유) →
`Inspire_Controller`. dex3 분기는 기존 튜플 유지(신규 인자는 default 사용, dex3 controller 미수신).

### 3.3 버그 수정
- `worker_deploy_dp.py`: `right_hand = chunk[:, 14+hd:14+2*hd]` (DEX3 hd=7 무변경, Inspire hd=6 정상화) + docstring.
- `worker_hand_ctrl.py`: deploy inspire clip `np.clip(hand_action, 0.0, 1.0)`.
- `robot_hand_inspire.py`: `is not None` 비교, 오기 주석 정정 (재작성에 포함).

---

## 3.4 그립 프로파일 UX (추가 — 사용자 피드백 반영)

사용자 요구: "상황별로 손가락 수 + 엄지 각도(특히 안쪽 회전=대향 각도)를 미리 만든 메뉴에서
골라 쓰기. 일일이 숫자로 적기 X." → 이름 붙인 프로파일 메뉴.

- `hand_control/inspire_grip_profiles.yaml`: `profiles.<name> = {grasp_fingers, close_depth,
  thumb_bend, thumb_yaw, grip_force, grip_speed}` + `default_profile`. 기본 5종:
  `full_oppose`(5지+엄지 대향), `tripod`(엄지+검지+중지), `pinch`(엄지+검지), `lateral`(엄지 측면),
  `hook`(엄지 미사용 4지). 파일에서 직접 보고/튜닝/추가.
- `main.py --grip-profile <name>` + `resolve_hand_shape(args)`: 우선순위
  **명시 플래그 > 프로파일 > 파일 default > fallback**. dex3 는 프로파일 무관(thumb_bend/yaw 만 의미,
  None→fallback). 해소된 값만 worker 로 전달(시그니처 불변).
- 엄지 모델 변경: `_grip_q` 에서 idx5(회전)=`thumb_yaw` 항상 적용(대향 각도), `thumb` 가
  grasp_fingers 에 포함되면 grasp 때 idx4=`thumb_bend` 로 굽고 open 때 펴짐. (기존 "엄지 완전 고정"
  → "엄지 회전=상황 고정 / 굽힘=grasp 참여" 로 정교화.)
- deploy 주의: deploy 의 손가락 q 는 정책이 직접 출력 → finger/thumb 항목 무의미, `grip_force/
  grip_speed` 안전 envelope 만 적용. "수집 때 쓴 --grip-profile 로 deploy" = envelope 일치.

## 4. 범위 밖 / 리스크
- DEX3 경로, record 스키마, GR00T deploy, hand-mode retargeting, tactile 저장: 미변경.
- `inspire_sdkpy` 가 현재 conda 인터프리터엔 미설치(소스는 `G1_1.7.../inspire_hand_sdk`).
  실로봇 실행 시 PYTHONPATH 필요 — 기존 코드도 동일 의존. vendoring 은 별도 논의.
- 실하드웨어 미연결 → 실증 테스트 불가. force/speed 기본값은 하드웨어에서 CLI 로 튜닝 가능하게 설계.

## 5. 검증 기준 (success criteria)
- `python -c "import ast; ast.parse(open(f).read())"` 전 파일 통과(문법).
- import 무결성: `from hand_control.robot_hand_inspire import Inspire_Controller` 구조상 호출부와 시그니처 일치.
- DEX3 회귀 없음: `worker_deploy_dp` hd=7 슬라이스 불변, dex3 spec 튜플/시그니처 불변.
- Inspire DP 슬라이스: hd=6 → left `14:20`, right `20:26` (수동 검산).
