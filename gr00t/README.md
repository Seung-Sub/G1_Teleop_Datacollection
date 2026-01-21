# Gr00t Policy Deployment

학습이 완료된 policy를 **Gr00t** 시스템에 배포하는 절차입니다.

## 디렉터리 구조

```
project-root/
├─ gr00t/
│  └─ result/                 # 학습 완료한 policy(체크포인트) 저장 위치
├─ worker/
│  └─ worker_deploy_policy.py    # self.model_path 설정 파일
```

## 배포 절차

### 1. 학습 완료한 Policy 저장

학습이 완료된 policy를 `gr00t/result/` 폴더에 저장합니다.

**파일 예시:**
- `gr00t/result/0801_apple/apple_checkpoint/checkpoint-100000`

### 2. 모델 경로 설정

`worker_deploy_policy.py`에서 `self.model_path`에 저장한 policy의 경로를 지정합니다.

```python
# worker_deploy_policy.py (예시)
class Worker:
    def __init__(self, ...):
        # ...
        self.model_path = "gr00t/result/0801_apple/apple_checkpoint/checkpoint-100000"
```

> **참고:** 상대/절대경로 모두 가능합니다. 경로 오타 및 파일명 불일치에 주의하세요.

### 3. Deploy

#### A. Conda **Teleop** 환경에서 Gr00t 모드 실행

```bash
# 예시 커맨드 (프로젝트별 실제 커맨드 사용)
conda activate Teleop
# Gr00t 모드로 Teleop 실행
# python teleop_main.py --mode gr00t
```

#### B. Conda **Gr00t** 환경에서 deploy 실행

```bash
# 예시 커맨드 (프로젝트별 실제 커맨드 사용)
conda activate Gr00t
# deploy 실행
# python worker_deploy_policy.py --deploy
```

#### C. UI 조작

1. **deploy model load** 완료 확인
2. UI에서 **Deploy** 버튼 클릭
3. **Start** 버튼 클릭하여 시작

## 체크리스트

- [ ] `gr00t/result/`에 policy 파일 존재 확인
- [ ] `worker_deploy_policy.py`의 `self.model_path`가 올바른 파일을 가리킴
- [ ] Conda 환경(패키지/드라이버) 일치 확인
- [ ] 런타임 로그에서 에러/경고 확인