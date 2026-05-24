# Meta Quest 3 — Linux USB 연결 가이드

이 문서는 **Quest 3 를 Linux PC (Ubuntu 22.04+) 에 유선(USB)으로 연결하여 우리
워크스페이스의 Vuer 서버를 헤드셋 안 브라우저에서 접속**하기 위한 단계.

## 0. Horizon Link 가 필요한가?

**Linux 에선 필요 없습니다.** Meta Horizon Link / Meta Quest Link 데스크톱 앱은
Windows / Mac 전용 PCVR 게이밍 앱입니다. Linux 에는 설치되지 않으며 헤드셋 내
브라우저가 "Meta Horizon Link 가 실행 중인지 확인하세요" 메시지를 띄워도 **무시해도
됩니다**.

우리 파이프라인은 헤드셋 안의 Quest Browser → HTTPS:8012 (Vuer 서버) → WebXR API
로 컨트롤러/HMD 입력을 직접 read 합니다. 데스크톱 PCVR 클라이언트는 무관.

---

## 1. PC 측 사전 준비 (Linux)

### 1-1. `adb` 설치
```bash
sudo apt-get update
sudo apt-get install -y adb     # jammy 이상 — 패키지명이 'adb' (옛 'android-tools-adb' 는 obsolete)
adb version    # 1.0.41 이상이면 OK
```

⚠️ Ubuntu 22.04 (jammy) 부터 `android-tools-adb` 패키지가 `adb` 로 rebrand 됨.
   `apt-get install android-tools-adb` 입력하면 "후보 (없음)" 으로 거절됨.
   또한 `apt-get update && apt-get install` 에서 외부 저장소 GPG/TLS 오류로
   update 가 exit 100 반환하면 `&&` 가 install 단계를 skip 하므로, 인덱스가
   이미 캐시에 있으면 update 없이 install 만 단독 실행해도 됨.

### 1-2. udev rule (root 없이 adb 가능)
```bash
# /etc/udev/rules.d/51-android.rules 생성
sudo tee /etc/udev/rules.d/51-android.rules > /dev/null <<'EOF'
# Meta / Oculus
SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0666", GROUP="plugdev"
EOF

# udev 재로드
sudo udevadm control --reload-rules
sudo udevadm trigger

# 사용자를 plugdev 그룹에 추가 (이미 들어가 있으면 무시됨)
sudo usermod -aG plugdev $USER
# 그룹 변경 적용을 위해 로그아웃 후 재로그인 (또는 newgrp plugdev)
```

### 1-3. adb-server 한 번 재시작 (rule 적용)
```bash
adb kill-server
adb start-server
```

---

## 2. Quest 3 측 사전 준비 (헤드셋 안)

Quest 3 가 adb 호스트로 인식되려면 **Developer mode + USB debugging** 이 ON 이어야
합니다. 처음 시도하는 경우:

### 2-1. Meta 개발자 계정 생성 + Quest 와 연결
1. PC 또는 폰 브라우저로 [dashboard.meta.com](https://dashboard.meta.com) 에 본인
   Meta 계정으로 로그인
2. **Developer Settings** 메뉴 → Developer organization 생성 (없다면)
3. Quest 모바일 앱 (Android/iOS) 에서 같은 계정으로 로그인 후 헤드셋 페어링

### 2-2. 헤드셋 안에서 Developer Mode 활성화
1. 헤드셋 착용 → Settings (설정)
2. **System** → **Developer** 메뉴 진입 (mobile app 으로 dev account 연결되어 있으면
   이 메뉴가 보임)
3. **USB Connection Dialog** ON
4. **USB Debugging** ON

### 2-3. USB-C 케이블 연결 + dialog 수락
1. **데이터 전송 가능한 USB-C 케이블**로 Quest 3 ↔ PC 연결
   - Quest 3 동봉 케이블은 보통 데이터 OK
   - 일부 USB-C 케이블은 충전만 가능 — `lsusb` 에서 Quest 가 안 보이면 케이블 의심
2. 헤드셋 안에 곧 다음 dialog 가 뜸:
   - **"이 컴퓨터에서 USB 디버깅을 허용하시겠습니까?"**
   - **"이 컴퓨터에서 항상 허용"** 체크 → **"허용"** 클릭
3. dialog 가 안 보이면: 케이블 뽑고 5초 대기 후 재연결

> Meta Horizon Link 가 PC 에 없다는 메시지는 다른 화면입니다. 그 화면은 닫고 위
> USB debugging dialog 를 찾으세요.

---

## 3. PC 에서 인식 확인

```bash
# 1) USB 레벨
lsusb | grep -iE 'oculus|meta|2833'
#  → 'Bus 003 Device 092: ID 2833:5013 Oculus VR, Inc. Quest 3'
#     product=5013 = adb debug 모드 (5012 면 아직 MTP 만 — dev mode 비활성)

# 2) adb 레벨
adb devices -l
#  → '<serial>  device  usb:3-8 transport_id:3'   ← 'device' 상태여야 OK
#  → 'unauthorized' 면 헤드셋 안 dialog 미수락 (재연결 후 dialog 수락)
#  → 빈 리스트면 USB debugging OFF 또는 케이블 데이터 라인 없음
```

상태 별 해법:

| `adb devices` 출력 | 의미 | 해법 |
|---|---|---|
| (빈 리스트) | adb daemon 이 device 못 봄 | USB debugging ON 인지, 케이블 데이터 OK 인지 확인. `adb kill-server && adb start-server` |
| `unauthorized` | USB debug ON 이지만 dialog 미수락 | 케이블 재연결 → 헤드셋 안 dialog 수락 |
| `offline` | 일시적 통신 끊김 | `adb kill-server && adb start-server` 또는 케이블 재연결 |
| `device` | 정상 ✓ | 다음 단계 |

---

## 4. Vuer 포트를 Quest 로 reverse

`adb reverse` 는 PC 의 포트를 Quest 의 localhost 로 매핑합니다. main.py GUI 의 'VR'
버튼이 자동 호출하지만 수동으로도 가능:

```bash
adb reverse tcp:8012 tcp:8012
adb reverse --list    # 'tcp:8012 tcp:8012' 가 보여야 함
```

이제 헤드셋 안 브라우저에서 `https://127.0.0.1:8012` 로 접속하면 PC 의 Vuer 서버에
연결됩니다.

> **자기 서명 cert (https) 경고** 가 헤드셋 브라우저에서 뜸:
> → "고급" 또는 "Advanced" → "안전하지 않은 사이트로 진입" 클릭.
> Vuer 가 사용하는 `cert.pem` / `key.pem` 은 작업 디렉터리에 있어야 함
> (없으면 `python -m vuer.cert` 또는 `openssl req -x509 ...` 로 생성).

---

## 5. 다음 단계

연결 확인되면:

```bash
cd /path/to/G1_Teleoperation
conda activate teleop

# 로봇 없이 Quest3 + IK 검증
python main.py --no-robot --vr-input controller --camera none \
               --hand dex3 --waist fixed --head off
```

병행으로 다른 터미널에서:

```bash
python scripts/verify_quest3.py --rate 2.0 --watch
```

상세 검증 절차는 `docs/DEPLOY_PRO4000.md` 의 "Quest3-only 동작 검증" 섹션 참고
(절차는 로컬 / pro4000 동일).

---

## 6. 자주 만나는 트러블슈팅

### `lsusb` 에 Quest 3 안 보임
- 케이블 의심 (충전 전용 케이블 가능성). 다른 USB-C 케이블 시도
- 헤드셋 화면이 켜져 있어야 함 (sleep 상태면 USB enumeration 안 됨)
- USB 허브를 거쳐 연결되어 있으면 PC 본체 포트에 직접 연결 시도

### `lsusb` 에 product id `2833:5012` 만 보이고 `5013` 안 보임
- USB debugging OFF 상태 (MTP 만 enable)
- 위 §2-2 의 developer mode + USB debugging 활성화 다시 확인

### `adb devices` 가 `no permissions (verify udev rules)`
- §1-2 의 udev rule 미적용. rule 작성 후 `udevadm control --reload-rules && udevadm trigger`
  실행 + 사용자가 plugdev 그룹에 들어가 있는지 (`groups | grep plugdev`)
- 그래도 안 되면 `sudo adb devices` 로 일단 진행 + udev rule 디버깅

### Vuer 페이지가 헤드셋에서 안 열림
- `adb reverse --list` 가 `tcp:8012 tcp:8012` 보여야 함. 안 보이면 `adb reverse tcp:8012 tcp:8012`
  재실행
- `curl -k https://127.0.0.1:8012` 가 PC 에서 응답 오는지 (Vuer 서버 살아있는지)
- cert.pem / key.pem 이 작업 디렉터리에 있는지

### 컨트롤러 입력이 잡히지 않음
- 헤드셋 안 설정 → Devices → Controllers 활성 / Hand tracking 비활성
- main.py 에 `--vr-input controller` 지정 (default 는 hand-tracking)
- Vuer 페이지에서 "Enter VR" 버튼 클릭해야 CONTROLLER_MOVE 이벤트 시작
- **vuer 0.0.60 client JS 가 hand-tracking 을 hardcode 로 요청**해 Quest 가 hand 우선 모드로
  떨어지면 controller 가 잡혀도 demote 됨. INSTALL.md §3-3 의 `python scripts/patch_vuer_xr.py disable`
  적용되어 있는지 `status` 명령으로 확인 (`PATCHED (handTracking disabled)` 표시).

### Vuer "Enter VR" 후 WebSocket 연결 실패
- vuer 0.0.60 의 client JS 가 WebSocket URL 에 port 를 누락해 Quest 가 wss://...:443 으로 연결 시도 → 실패.
  동일 patch script (`patch_vuer_xr.py disable`) 가 port 누락 bug 도 함께 수정.
