# 인수인계 — 학습 머신에서 이어받는 사람에게

작성 2026-08-06. 여기까지는 **출장지 노트북(Windows, GPU 없음)** 에서 만들었다.
그래서 구조·검증은 끝났지만 **실데이터로는 한 번도 안 돌았다.**

---

## 1. 한 줄 요약

레몬 in-hand 회전 + 낙하 회복용 visuo-tactile diffusion policy 프레임워크.
config 하나로 `rgb 인코더 × tactile 인코더 × action 생성기 × 주파수 비율` 을 갈아끼운다.
**학습 파이프라인은 합성 데이터로 end-to-end 검증 완료. 남은 건 실데이터를 물리는 것.**

---

## 2. 지금 무엇이 검증됐나 (그리고 무엇이 안 됐나)

| 항목 | 상태 | 근거 |
|---|---|---|
| 모듈 조합 shape·backward·sample | ✅ 50/50 | `tests/test_shapes.py` |
| config 10개 조립 + optimizer 스텝 | ✅ 14/14 | `tests/test_configs.py` |
| 로더 의미(주파수 비율·누출·상수채널·RGB·체크포인트) | ✅ 11/11 | `tests/test_data.py` |
| 기준선 재현 | ✅ | `10_baseline_refer` = refer/dp_unet **2,469,200 파라미터 일치** |
| 학습 루프(preflight/smoke/full·EMA·resume) | ✅ | 합성 HDF5 위에서 end-to-end |
| **실데이터 학습** | ❌ | **한 번도 안 함** |
| **실기 배포(`run.py` 연결)** | ❌ | 미구현 |
| RGB decode 가 20Hz 예산에 드는지 | ❌ | 실기에서만 측정 가능 |

⚠️ 합성 데이터는 `tools/make_fake_h5.py` 가 만든다. 레코더와 **형식은** 같지만(vlen JPEG +
`41_rgb_index` 간접참조 + 카메라 attrs) **내용은 가짜**다. 형식 버그는 잡히지만
"진짜 신호의 통계" 관련 문제는 여기서 안 잡힌다.

---

## 3. 도착하면 이 순서로

### 3-1. 데이터 판정 (10초)

```bash
python3 tools/inspect_h5.py /home/js/Desktop/vive_franka_teleop/record/logs
```

`h5py`+`numpy` 만 있으면 돈다. 다섯 가지가 한 번에 나온다:

**⚠️ 먼저 보는 것: `Demo_*` 그룹이 있는가.** 없으면 대상 레코더 포맷이 아니다
(구버전은 `demo0` 소문자 + `(N,1,C)` 축 + 1000Hz + Paxini 없음 — 이 머신의
`hdf5/exp2_20260311_..._tennisball.h5` 가 그 예다). 이건 "RGB 가 없다"와 **다른 문제**이고
재수집이 아니라 레코더 버전을 확인해야 한다. `inspect_h5.py` 가 이제 이걸 먼저 판정한다.

| 판정 | 결과에 따라 |
|---|---|
| **RGB 가 있나** | 없으면 `01`~`09` 전부 못 쓴다 → **`configs/11_no_rgb.yaml` 을 바로 돌린다**(state/tactile 분리 + 손가락 4토큰, `10` 과 타이밍 동일). RGB 재수집(`--rgb-on`)은 별도로 잡는다 |
| **`20_paxini_ft` 가 몇 부위 × 몇 채널인가** | `inspect_h5.py` 의 `[레이아웃 추정]` 이 블록내/블록외 상관을 뽑아 준다. `n_part` 가 틀리면 **조용히 엉뚱한 손가락을 학습한다** |
| `21_paxini_raw` 살아있는 채널 수 | 10% 미만이면 센서 일부만 동작한 것 |
| `31_fruit_quat` 이 항등인가 | 살아있으면 **회전 위상 관측이 생긴다** → `refer/BASELINES.md` 의 "POMDP" 진단이 뒤집힌다. state keys 에 추가 검토 |
| 카메라 intrinsics | 있으면 나중에 fingertip 3D→2D 투영 grounding 가능 |
| RGB 동결 구간 | 길면 그 구간 데이터는 시각적으로 무의미 |

**RGB 저장은 레코더의 `--rgb-on` opt-in 이다**(`refer/ros2_hdf5_recorder.py:384`).
이게 이 프로젝트 최대의 단일 리스크다.

### 3-2. 파이프라인 증명 (5분)

```bash
python3 train.py --config configs/01_anchor.yaml \
  --data /home/js/Desktop/vive_franka_teleop/record/logs --mode preflight
```

데모 표·정규화 통계·shape 검증·ETA 가 나오고 모델은 안 돌린다. 여기서 걸리는 건 대개
`config` 의 `keys` 와 실제 HDF5 키 불일치다 — 오류 메시지가 어느 키가 몇 차원인지 알려준다.

그다음 `--mode smoke` 로 몇 스텝, 통과하면 `--mode full`.

### 3-3. 데이터 선별을 켤지 결정

`configs/_base.yaml` 의 `data.clamp_spec: null` 은 **폴더의 모든 데모를 쓴다**는 뜻이다.
`refer/dp_config.py` 의 `CLAMP_SPEC` 에는 사람이 손으로 적어둔 가위질 목록(어느 데모를
어디까지 쓸지, 어느 것은 버릴지)이 있다. 그 22개 데모를 재현하려면 그걸 옮겨와야 한다.
`load_demos` 가 같은 형식(`{파일명: {데모idx: None|int|"DEAD"|"DELETE"}}`)을 받는다.

### 3-4. 실험 10개

`configs/01_anchor.yaml`(resnet18 + lstm + DP-CNN + 1:1)에서 **한 축씩** 바꾼 게 `02`~`08`,
`09`는 조합(결과 보고 고쳐 쓸 것), `10`은 refer/ 정확 재현이다.

**`10_baseline_refer` 를 반드시 같이 돌려라.** 못 이기면 나머지는 의미가 없다.

---

## 4. 미해결 — 1524-D 촉각

`21_paxini_raw` 가 어떤 구조인지 모른다. **이게 촉각 인코더 선택을 전부 좌우한다.**

같은 벤더(Paxini)를 쓴 3DTacDex(`github.com/tianhaowuhz/3dtacdex`, MIT)의 레이아웃:

```python
POINT_PER_SENSOR = 15        # 센서당 15 taxel
FORCE_DIM_PER_POINT = 3      # taxel당 3축
PAXINI_LEAPHAND = {thumb, index, middle, ring}   # 손가락당 tip + pulp
# 4 x 2 x 15 x 3 = 360
```

우리는 **1524** 다. `1524 = 3 × 508`, `508 = 4 × 127`(127은 소수) → 단순 격자로 안 떨어진다.
헤더·상태 바이트가 섞였을 가능성이 높다. **KIST 쪽 `/paxini/right/raw` 퍼블리셔 코드를
봐야 확정된다.**

- taxel별 `(Fx,Fy,Fz)` + taxel 3D 위치를 복원할 수 있으면 → `tactile.taxel_cnn`, CoP,
  FK 앵커링이 열린다. 3DTacDex 는 taxel 좌표(`PAXINI_DP_ORI_COORDS`)와 148KB 인코더
  가중치까지 공개해 뒀다.
- 불투명한 덩어리면 → `20_paxini_ft` 12-D 요약에 머무른다(현재 config 가 이 상태).

`tactile.taxel_cnn` 은 레이아웃이 안 맞으면 **생성 시점에 거부**한다.
틀린 재배열은 조용히 학습되고 조용히 나쁘기 때문이다.

---

## 5. 설계 결정과 그 근거

vault(개인 연구 wiki 267개 소스)와 공개 논문 서베이를 교차검증해서 나온 것들이다.
**취향이 아니라 측정된 결과에 근거한다.**

| 결정 | 근거 |
|---|---|
| 모달리티별 인코더 분리 | naive concat 반복 실패: 3DTacDex 순진한 taxel GNN **4개 태스크 전부 0%** · CONTACT "naive fusion 이 단독보다 나쁘다" · ManiFeel 시각우세 태스크 **43.7%→22.0%** · vault 에 T-Rex/ForceVLA2/GeoProp/FTP-1/CoorDex 5건 |
| flatten raw taxel 을 기본으로 안 씀 | Beyond Binary(16-DoF 손 + 3축 taxel array): CoP 0.78 > binary 0.53 > **raw flatten 0.48** |
| 접촉 게이팅(`fusion: gated`) | M2-ResiPolicy·Dream-Tac CASA·FoAR 3개 논문 독립 수렴 |
| frozen backbone 기본 | 22 데모에 11M 파인튜닝은 위험. ManiFeel: **scratch ResNet-18 이 촉각 foundation model 들과 대등** |
| flow 를 기본으로 안 씀 | ManiFeel 이 같은 조건에서 DP·Equivariant DP 와 비교, flow 가 둘 다에 뒤짐 |
| best = 홀드아웃 액션 MAE | `refer/BASELINES.md` 규약. diffusion/flow/bc 를 같은 자로 재야 비교 가능 |
| 상수 채널 std=1 | refer/ 의 `std.clip(1e-6)` 은 배포 때 10^6 배 폭발 위험 |

### 참고할 만한 외부 코드 (검증함, 2026-08-06 기준 실재)

| 레포 | 왜 |
|---|---|
| [tianhaowuhz/3dtacdex](https://github.com/tianhaowuhz/3dtacdex) MIT | **Paxini + 16-DoF 손 + ~30 demo + diffusion.** 유일하게 하드웨어가 맞는 공개 코드. 인코더 가중치 148KB 커밋됨 |
| [kingchou007/adaptac-dex](https://github.com/kingchou007/adaptac-dex) MIT | 위의 후속. **force-guided cross-attention** + future-force 예측 auxiliary loss(50%→90%) |
| [AdeeshDesai/CONTACT](https://github.com/AdeeshDesai/CONTACT) | config 가 우리 기준선과 동일. **force field 표현이 gel 이미지를 이김(70% vs 30%)** |
| [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) MIT | vision 레시피 정본(ResNet18 + GroupNorm + spatial softmax) |
| [gist-ailab/ManipForce](https://github.com/gist-ailab/ManipForce) | 사용자 본인 연구. `FTEmbed`(learnable α residual)와 양방향 cross-attention 을 여기서 이식 |

**FARM 은 논문이 오픈소스라고 주장하지만 레포가 404 다.** 3D-ViTac 학습 코드는 스텁이다.

---

## 6. 함정 모음 (겪은 것들)

- **config 병합**: `obs_spec` 과 `*_kwargs` 는 deep merge 가 아니라 **통째 교체**다.
  안 그러면 대조군을 만들 수 없고(모달리티를 못 뺀다), 모듈 바꿀 때 부모 kwargs 가 따라와 터진다.
- **h5py + DataLoader**: `File` 객체는 fork 를 넘어 공유하면 안 된다. `data.py` 는 워커
  PID 별로 지연 오픈한다. `num_workers` 를 올릴 때 이 구조를 깨지 말 것.
- **Windows 콘솔**: cp949 기본이라 한글/이모지 출력에서 죽는다. 스크립트마다 stdout 을
  utf-8 로 reconfigure 해 뒀다. Linux 에서는 무해.
- **DINOv2 입력**: patch 14 의 배수여야 한다(224 = 14×16). 첫 실행에 가중치 다운로드가
  필요하니 오프라인이면 `~/.cache/torch/hub` 를 미리 채울 것.
- **체크포인트 142MB**: frozen backbone 이 policy/ema/optimizer 3벌로 저장된다.
  frozen 파라미터를 제외하면 크게 준다 — 아직 안 고침.
- **주파수 비율**: `1:2`, `1:3` 이 정수로 떨어지려면 rgb stride 가 6의 배수여야 한다.
  그래서 rgb 는 stride 6(16.7Hz), action 은 stride 5(20Hz, refer/ 검증값)로 **따로** 간다.
  `vtdp.config.timing_table(cfg)` 로 격자를 눈으로 확인할 것.

---

## 6-1. 2026-08-06 2차 점검에서 고친 것 (실측 근거 포함)

디버깅 전 상태에서 찾아 고친 것들. 전부 회귀 테스트가 붙었다.

| 고친 것 | 증상 | 근거 |
|---|---|---|
| **frozen + BN→GN** | 시각 분기가 거의 상수 출력 | 이미지-의존 성분 72.8%→15.3%, 이미지간 코사인 0.471→0.976. `norm: auto` 로 frozen 이면 BN 유지 |
| **촉각이 손가락을 섞음** | `Linear(12,256)` 이 4손가락×3축을 첫 층에서 합침 | `n_part` + 부위 임베딩. lstm 은 부위 간 누출 0, transformer 는 4배 선택성 |
| **촉각 시간축 붕괴** | `return_seq=False`/`use_cls=True` 가 기본이라 T→1토큰 | 기본을 시간 보존으로 바꿈 (n_tokens = n_part×T) |
| **`cross_attn` pooled = mean** | cross-attention 으로 만든 토큰 구조를 평균이 지움 | `pool: flatten` 기본 |
| **LoRA 없음** | DINOv2 는 완전 frozen 뿐 | `lora_rank` 추가. frozen 인데 LoRA 면 `no_grad` 를 벗겨야 학습된다(안 벗기면 조용히 안 배움) |
| **LoRA OOM** | 활성값을 전부 들고 있어야 함 | `grad_ckpt: true` — 6.81→2.06 GiB, step +30% |
| **`lr_sched` 미저장** | 재개 시 LR 이 warmup 부터 다시 | 저장·복원 추가. 중단/재개 LR 이 무중단과 일치하는지 확인함 |
| **`prefer_weights` 미기록** | best.pt 만으로 raw/ema 를 못 고름 → `run.py --weights auto` 깨짐 | `best_metric` 기록 |
| **비원자적 저장** | 142MB 쓰다 죽으면 체크포인트 손상 | tmp → `os.replace` |
| **eval 노이즈** | `generator=None` 이라 best 가 운으로 뽑힘 | `eval_seed: 7` 배치별 고정 |
| **eval 비용** | 매 에폭 홀드아웃 전체 ×2 = 에폭 시간의 56% | `eval_every: 5`, `eval_n: 768` 균등 추출 |
| **backbone LR** | 사전학습 가중치를 head 와 같은 3e-4 로 밀어버림 | `lr_backbone_scale: 0.1` 파라미터 그룹 분리 |
| **RGB 미래 누출** | `41_rgb_index == -1` 구간이 미래 프레임을 당겨옴 | 과거만 참조 + 시작 구간 윈도우 제외. 합성 데이터가 -1 을 안 만들어 테스트가 못 잡던 경로 |
| **`n_held_out: 0`** | `max() arg is an empty sequence` | 이유를 설명하며 거부 |
| **SIGTERM** | 프로세스 그룹 신호면 worker 가 먼저 죽어 crash 경로로 떨어짐 | worker 사망을 정상 종료로 처리, 저장 중 워치독 차단 |
| **`alpha_bar` 가 plain 속성** | 스텝마다 CPU→GPU 복사 | non-persistent buffer |

미해결로 남긴 것: `mask` 계약은 `to_device` 가 이제 전달하지만 **dataset 이 mask 를 만들지
않는다**(modality dropout 은 policy 안에서 생성). `21_paxini_raw` 1524-D 구조도 그대로 미해결.

## 7. 다음 작업 (우선순위)

0. **`tools/inspect_h5.py` 로 촉각 레이아웃 확정.** `n_part: 4`(4손가락×3축)를 가정해 뒀다.
   실제가 `N × 6` 이면 `obs_spec.tactile.shape` 와 `n_part` 를 같이 고쳐야 한다.
   **틀리면 조용히 엉뚱한 손가락을 학습한다** — 학습 전에 반드시 확인.
1. **`run.py` 배포 연결** — `refer/run.py` 가 ROS2 실기 루프를 이미 갖고 있다.
   체크포인트에서 정책+정규화를 복원하는 경로만 새로 붙이면 된다(`policy_from_ckpt` 대응).
   `prefer_weights` 는 이제 체크포인트에 있으니 `--weights auto` 가 바로 동작한다.
2. **RGB decode 지연 측정** — `refer/bench_latency.py` 를 확장. 20Hz(50ms) 예산 안에
   JPEG decode + resize + ResNet forward + DDIM 10스텝이 들어가야 한다. 안 들어가면
   `infer_steps` 를 줄이거나(2스텝도 보고된 바 있음) `exec_horizon` 을 늘린다.
3. **future-force 예측 auxiliary loss** — AdapTac 진단상 촉각 분기가 무시되는 걸 막는
   가장 효과가 큰 장치(50%→90%). 촉각 branch 에 n스텝 뒤 힘을 예측하는 헤드를 달고
   `L = L_π + α·L_ffp`.
4. **평가 rubric** — 성공률만으로는 22 데모 규모에서 정책을 못 가른다(Clopper-Pearson
   CI 20~30%p). **"파지/90도 회전/낙하/회수"** milestone rubric 을 같이 기록할 것.
   "미끄러졌으나 회수" 가 정확히 이 태스크의 촉각 고유 사건인데 성공률엔 안 잡힌다.
5. CoP 표현 — 1524-D 해독 후.

---

## 8. 환경

```bash
python -m venv .venv
.venv/bin/pip install torch torchvision numpy h5py pyyaml pillow
python tests/test_shapes.py && python tests/test_configs.py && python tests/test_data.py
```

작성 시점 검증 환경: Python 3.11, torch 2.13.0+cpu, torchvision, numpy 2.4.6, h5py 3.16,
pyyaml 6.0.3, pillow 12.2. GPU 에서는 `--device cuda` (기본값이 자동 감지).
