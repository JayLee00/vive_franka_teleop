# Diffusion Policy — 레몬 in-hand 회전 (KISTAR 오른손 16관절)

`record/logs` 의 HDF5 데모로 저차원 Diffusion Policy 를 학습하고 실기 배포한다.
아키텍처는 `~/Franka_Dual_Arm_PtoP/R_Franka_KISTAR_Hand/diffusion` (전구 돌리기)을 이식했다.

## 스펙

| | |
|---|---|
| obs (46) | `03_hand_j_pos`(16) + `06_hand_j_kin`(12) + `20_paxini_ft`(12) + `30_fruit_pos`(3) + `32_fruit_size`(3) |
| action (16) | `04_hand_j_tar` 절대 관절 타겟 [count] (1 count = π/8192 rad) |
| 시간축 | 원본 100Hz → **stride 5 = 20Hz**. T_obs=2, T_pred=16(0.8s), T_exec=8(0.4s) |
| 모델 | MLP obs encoder → global_cond(256) → FiLM 1D U-Net 디노이저 |
| 확산 | Cosine DDPM T=100 학습 / DDIM 10스텝 결정적 추론 |

## 파일

| 파일 | 역할 |
|---|---|
| `dp_config.py` | obs/action 스펙, 시간축, 관절·속도 한계, **데모별 클램프 표** |
| `dp_data.py` | HDF5 로딩 → 클램프 → 과일 이상치 위생처리 → dilated 윈도우 → 데모 단위 분할 |
| `dp_model.py` | U-Net / DiT 디노이저, BC 기준선, DDPM·DDIM, EMA |
| `train.py` | preflight / overfit / smoke / full |
| `eval_rollout.py` | 홀드아웃 open-loop rollout, 관절별 MAE, 플롯 |
| `sweep.py` | 설정 비교 (정규화·모델·알고리즘) |
| `run.py` | ROS2 실기 배포 (청킹·제어 Hz·안전 가드) |
| `BASELINES.md` | SOTA 후보 정리 + 실측 비교 + 다음 수 |

## 학습

```bash
python3 train.py --mode preflight              # 클램프 표·shape·NaN·ETA 만
python3 train.py --mode overfit                # 배치1 과적합 → 파이프라인 증명
python3 train.py --mode smoke --smoke_minutes 5
python3 train.py --mode full --epochs 200 --out_dir runs/dp_lemon
```

- 정규화 통계는 **train split 프레임만**으로 계산하고 `obs_norm.npz`/`act_norm.npz` +
  **체크포인트 내부**에 함께 저장한다. 배포 시 `run.py` 는 체크포인트에서 읽으므로
  전처리 불일치가 생길 수 없다.
- 체크포인트: `best.pt`(홀드아웃 **액션 MAE** 최저) / `last.pt`(auto-resume) /
  `ep####.pt`(주기) / `crash.pt`(예외·SIGTERM). EMA 가중치 항상 포함.
- `best.pt` 선택 기준은 노이즈 MSE 가 아니라 **DDIM 으로 실제 액션을 뽑은 count 단위 MAE**다
  (알고리즘 간 비교 가능하고 배포 성능에 가깝다).
- 재시작하면 `last.pt` 에서 자동 재개 (`--resume none` 으로 끔). 시드 고정.

## 평가

```bash
python3 eval_rollout.py --ckpt runs/dp_lemon/best.pt                     # raw vs ema
python3 eval_rollout.py --ckpt runs/dp_lemon/best.pt --temporal_ensemble  # TE 효과
```
`runs/*/eval/` 에 `eval_summary.json`, `per_joint_mae_*.png`,
`error_vs_horizon_*.png`, `rollout_<demo>_*.png`.

> **한계**: 시뮬레이터가 없어 관측을 GT 궤적에서 읽는다. 액션 예측 정확도이며
> 태스크 성공률이 아니다. 최종 판단은 실기.

## 배포

```bash
source /opt/ros/humble/setup.bash && source ~/franka_ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0

# 0) 글러브 텔레옵을 반드시 먼저 종료 (q_target 입구는 하나)
pkill -f glove_teleop.py

# 1) 발행 없이 관측·추론·지연만 확인
python3 run.py --ckpt runs/dp_lemon/best.pt --dry_run

# 2) 인게이지 후 실행
ros2 topic pub -1 /dp/enable std_msgs/Bool "{data: true}"
python3 run.py --ckpt runs/dp_lemon/best.pt

# 정지
ros2 topic pub -1 /dp/enable std_msgs/Bool "{data: false}"
```

### 조절 노브

| 인자 | 기본 | 의미 |
|---|---|---|
| `--control_hz` | 20 (학습값) | 정책 추론·액션 소비 주기. 바꾸면 학습과 시간축이 달라진다 |
| `--publish_hz` | 100 | 실제 타겟 발행 주기. 정책 타겟 사이를 선형보간 → 스텝 없이 매끄럽게 |
| `--exec_horizon` | 8 | 추론 1회당 실행할 액션 수. 작을수록 반응 빠름·추론 부하 큼 |
| `--pred_horizon` | 16 | 예측 길이 (학습값 유지 권장) |
| `--temporal_ensemble` | off | 매 틱 추론 + 겹치는 예측 지수가중 평균 (ACT 방식) |
| `--ddim_steps` | 10 | 적으면 빠르고 거칠다 |
| `--weights` | auto | `best.pt` 가 기록한 더 좋은 쪽 (raw/ema) |

### 안전 가드

| 가드 | 동작 |
|---|---|
| 관절 한계 | 체크포인트의 `JOINT_LIMITS`(= `tools/glove_teleop.py` 표)로 클램프 |
| 속도 한계 | `--max_rate_cps` 기본 12000 count/s = 학습 데이터 20Hz 틱당 실측 p99.9(600) |
| 시작 램프 | `--ramp_sec` 2초 동안 현재 손 자세 → 정책 타겟 (튐 방지) |
| 워치독 | obs 가 `--stale_sec`(0.3s) 이상 낡으면 새 타겟 발행 중단 → 제어 PC 가 홀드 |
| 인게이지 | `/dp/enable` 이 True 여야 발행. False 로 즉시 정지 |
| 중복 발행 감지 | `q_target` 퍼블리셔가 2개 이상이면 2초마다 에러 로그 |
| `--dry_run` | 계산만, 발행 없음 |

진단은 `/dp/debug` (Float64MultiArray):
`[enabled, fruit_ok, infer_ms, act_buf, |Δ|max, clamp_joint, clamp_vel, stale]`
