# kist-vtdp-wrapper

KISTAR 16관절 손 + Franka, 레몬 in-hand 회전 + 낙하 회복 태스크용
**visuo-tactile diffusion policy** — 모듈을 config 로 갈아끼우는 프레임워크.

`refer/` 의 저차원 Diffusion Policy 를 기준선으로 두고, **모달리티별 인코더 · fusion ·
action head 를 각각 독립 축으로** 분리해 조합 실험이 곱집합으로 나오게 한다.

> **이어서 작업한다면 [`CLAUDE.md`](CLAUDE.md) 를 먼저 읽을 것.**
> 상세한 인수인계(무엇이 검증됐고, 무엇이 미확인이고, 어디부터 손대야 하는지)는
> [`docs/HANDOFF.md`](docs/HANDOFF.md) 에 있다.

---

## 지금 상태

| | |
|---|---|
| 조합 shape 테스트 | **50/50 통과** (`tests/test_shapes.py`) |
| config 테스트 | **19/19 통과** (`tests/test_configs.py`) |
| 로더 테스트 | **13/13 통과** (`tests/test_data.py`) — 합성 HDF5 위에서 |
| 기준선 재현 | `dp_unet` 2,469,200 — **테스트가 강제한다.** `dp_dit` 는 `max_len: 64` 를 줘야 5,964,688 (기본값은 pos-emb 48슬롯이 없어 5,952,400) |
| 학습 루프 | ✅ preflight / smoke / full · EMA · resume(LR 스케줄 포함) · best=홀드아웃 MAE · SIGTERM 정상 종료 |
| 실데이터 학습 | ⏳ 데이터만 있으면 바로 — 실제 HDF5 미확인 |

---

## ⚠️ 연구실 도착하면 이것부터

시각 분기 전체가 여기에 걸려 있다. 레코더의 RGB 저장은 **`--rgb-on` opt-in**
([refer/ros2_hdf5_recorder.py:384](refer/ros2_hdf5_recorder.py))이라
**기존 22개 데모에 RGB 가 아예 없을 수 있다.**

```bash
python3 tools/inspect_h5.py /home/js/Desktop/vive_franka_teleop/record/logs
```

h5py + numpy 만 있으면 돈다(torch·ROS 불필요). 한 번에 5가지가 판정된다:

1. **RGB 가 있나** — 없으면 `03/04` config 는 재수집 전까지 못 쓴다
2. **`21_paxini_raw`(1524ch) 중 몇 채널이 살아있나**
3. **`31_fruit_quat` 이 항등쿼터니언인가 실제 각도인가** — 살아있다면
   `BASELINES.md` 의 "과일 각도가 없어 POMDP" 진단이 바뀐다
4. **카메라 내부파라미터가 있나** — 3D→2D 투영 기반 공간 grounding 가능 여부
5. **RGB 프레임이 얼어붙은 구간이 있나**

### 그 다음 — 1524-D 가 무엇인지 해독

이게 촉각 인코더 선택을 전부 좌우한다. 참고로 같은 벤더를 쓰는 3DTacDex 의 레이아웃은:

```python
POINT_PER_SENSOR = 15        # 센서당 15 taxel
FORCE_DIM_PER_POINT = 3      # taxel당 3축
# 4손가락 x (tip + pulp) x 15 x 3 = 360
```

우리는 1524 다. `1524 = 4 × 381 = 3 × 508`, 그리고 `508 = 4 × 127`(127은 소수)이라
**단순한 격자로 안 떨어진다.** 헤더·상태 바이트가 섞여 있을 가능성이 높다.
KIST 쪽 `/paxini/right/raw` 퍼블리셔 코드를 봐야 확정된다.

- **taxel별 (Fx,Fy,Fz) 와 taxel 의 3D 위치를 복원할 수 있으면** → `tactile.taxel_cnn`,
  CoP, FK 앵커링까지 열린다
- **불투명한 덩어리면** → 12-D F/T 요약에 머물러야 한다

`tactile.taxel_cnn` 은 레이아웃이 안 맞으면 **생성 시점에 거부**한다.
틀린 재배열은 조용히 학습되고 조용히 나쁘기 때문이다.

---

## 구조

```
vtdp/
  registry.py    이름 → 생성자. Hydra _target_ 대신 — 오류 메시지가 훨씬 낫다
  encoders.py    state / tactile / vision 인코더. 전부 (B, n_tokens, D) 를 낸다
  fusion.py      passthrough / concat / cross_attn / gated
  heads.py       denoiser(unet1d·dit·transformer) × objective(diffusion·flow·bc)
  policy.py      config → 조립
  config.py      YAML 로드·상속·검증
  data.py        HDF5 → 모달리티별 윈도우 · 정규화 · RGB 디코드
train.py         학습 루프 (preflight / smoke / full)
configs/         실험 config (아래 표)
tools/
  inspect_h5.py   데이터에 뭐가 들어있는지 판정
  make_fake_h5.py 레코더와 같은 형식의 합성 데이터 (실데이터 없이 검증)
tests/
  test_shapes.py  조합 곱집합 shape·backward·sample 검사
  test_configs.py configs/*.yaml 전부 조립 + optimizer 스텝까지
  test_data.py    로더 의미 검증 (주파수 비율·누출·상수채널·RGB·체크포인트)
docs/
  SHAPES.md      ★ 텐서 계약. 모듈 추가 전에 읽을 것
refer/           원본 참고 구현 (수정하지 않는다)
```

### 축 4개

| 축 | 선택지 |
|---|---|
| **1. rgb 인코더** | `resnet18`(ImageNet, **파인튜닝** + GroupNorm + spatial softmax) · `dinov2`(ViT-S/14 + **LoRA**) · `smallcnn`(scratch, CPU 테스트용) |
| **2. tactile 인코더** | `lstm` · `transformer` (둘 다 `n_part` 로 **손가락 축 보존**) · `conv1d`(ManipForce FTEmbed) · `taxel_cnn`(1524-D 해독 후) |
| **3. action 생성기** | denoiser `unet1d`(DP-CNN) · `transformer`(DP-Transformer, cross-attn) · `dit`(adaLN-Zero) × objective `diffusion`(DDIM) · `flow` · `bc` |
| **4. 주파수 비율** | 모달리티별 `stride` — rgb:tactile = 1:1 / 1:2 / 1:3 |

**축 1·2 는 두 가지 함정이 있다** (`docs/SHAPES.md` 2.2.1·2.2.2 에 측정치와 함께 적어 뒀다):

- **시각**: `frozen: true` + BN→GN 치환은 사전학습 통계를 버려서 **feature 가 거의 상수가
  된다**(이미지-의존 성분 72.8%→15.3%). `norm: auto` 가 파인튜닝이면 GN, frozen 이면
  BN(eval 고정)을 고른다. DINOv2 LoRA 는 `grad_ckpt: true` 필요(6.81→2.06 GiB).
- **촉각**: `n_part` 없이 12-D 를 그대로 Linear 에 넣으면 4손가락 × 3축이 첫 층에서
  섞인다. `n_part: 4` + `return_seq: true` 로 **부위축과 시간축을 둘 다** 남긴다.
  그리고 `unet1d`/`dit` 은 pooled 만 보므로 `pool: flatten` 이어야 그 구조가 살아 넘어간다.

**주파수 비율은 `obs_spec.<k>.stride` 로 표현한다.** lookback = `(horizon-1) x stride` 를
모든 모달리티에서 같게 두고 stride 만 나누면 주파수만 바뀐다. 다르면 히스토리 길이를
바꾼 것이고, refer/ 스윕에서 히스토리 확장(T_obs 4/8/16)은 전부 5~6% 나빴다.

`vtdp.config.timing_table(cfg)` 가 격자를 펼쳐 보여주고 lookback 이 어긋나면 경고한다.

fusion 은 축이 아니라 조합 슬롯으로 남겼다: `concat` · `cross_attn`(ManipForce) ·
`gated`(접촉 게이팅) · `passthrough`(기준선 재현용, 파라미터 0).

---

## config

```bash
python tests/test_configs.py                       # 전부 조립되는지
python -c "from vtdp.config import load_config, timing_table as t; print(t(load_config('configs/08_freq_1to3.yaml')))"
```

| # | config | 바꾸는 축 | 값 |
|---|---|---|---|
| 01 | `01_anchor` | — | resnet18(파인튜닝) + lstm(4손가락) + DP-CNN + 1:1 (**기준 조합**) |
| 02 | `02_rgb_dinov2` | 축1 rgb | → DINOv2 ViT-S/14 + LoRA(r=8) |
| 03 | `03_tac_transformer` | 축2 tactile | → transformer (부위×시각 토큰) |
| 04 | `04_gen_dp_transformer` | 축3 생성기 | → DP-Transformer |
| 05 | `05_gen_dit` | 축3 생성기 | → DiT |
| 06 | `06_gen_flow` | 축3 생성기 | → flow matching |
| 07 | `07_freq_1to2` | 축4 주파수 | tactile 33.3Hz (1:2) |
| 08 | `08_freq_1to3` | 축4 주파수 | tactile 50Hz (1:3) |
| 09 | `09_best_combo` | 조합 | 01~08 결과 보고 **고쳐 쓸 것** (현재는 가설) |
| 10 | `10_baseline_refer` | — | refer/ 정확 재현 (2,469,200 파라미터 일치) |
| 11 | `11_no_rgb` | — | **RGB 가 없을 때의 진입점.** `10` 과 타이밍·키가 같고 모달리티 분리 + 손가락 4토큰만 다르다 → 그 차이가 곧 "분리가 값을 하는가"의 답 |

01~08 은 **한 번에 한 축만** 바꾼다. 홀드아웃이 3 데모뿐이라 두 축을 동시에 바꾸면
차이의 원인을 짚을 수 없다.

상속은 `_base_` 체인(최대 8단계, 순환 검출)이고 두 가지 예외 규칙이 있다:

- **`obs_spec` 은 통째 교체** — merge 하면 자식이 부모의 모달리티를 *뺄 수 없어서*
  대조군을 만들 수 없다
- **`*_kwargs` 도 통째 교체** — kwargs 는 자기 모듈 선택에 종속이라,
  fusion 을 바꿨는데 부모 kwargs 가 남으면 생성자가 터진다

override:
```bash
--override model.fusion=gated model.modality_dropout=0.2
--override model.denoiser=dit model.denoiser_kwargs={}         # 모듈을 바꾸면 kwargs 도 비운다
--override model.fusion_kwargs='{gate_on: tactile, floor: 0.1}'  # 없는 하위 키는 못 만든다
```
없는 경로는 즉시 거부되고 가능한 키를 알려준다.
`*_kwargs` 는 통째 교체라 **모듈만 바꾸면 부모 kwargs 가 남아 거부된다** — 위처럼 같이 준다.

---

## 설계 근거 (증거)

이 구조는 취향이 아니라 vault 267개 소스 + 공개 논문 서베이 교차검증에서 나왔다.

- **naive concat 은 반복적으로 실패했다.** 3DTacDex: 순진한 taxel GNN → **4개 태스크 전부 0%**.
  CONTACT: "TacRGB+TacFF naive fusion 이 각각 단독보다 나쁘다". ManiFeel: 시각 우세 태스크에서
  **43.7% → 22.0%**. vault 는 같은 병리를 T-Rex·ForceVLA2·GeoProp·FTP-1·CoorDex 5건으로 모아뒀다.
  → 그래서 `10_baseline_refer` 를 **반드시 같이** 돌린다. 못 이기면 나머지는 의미가 없다.
- **flatten 한 raw taxel 은 1-bit 접촉 신호보다도 나빴다** (Beyond Binary, 16-DoF 손 +
  3축 taxel array: CoP 0.78 > binary 0.53 > **raw flatten 0.48**). 그 arm 이 현재 baseline 의 처리다.
- **촉각 분기는 자기 일을 주지 않으면 무시된다.** AdapTac 진단: attention 만으로는
  modality imbalance 가 안 고쳐졌고 **future-force 예측 auxiliary loss** 가 고쳤다(50%→90%).
  → 미구현. 다음 작업 후보 1순위.
- **접촉 게이팅은 3개 논문이 독립 수렴했다** (M2-ResiPolicy 이진 threshold · Dream-Tac CASA ·
  FoAR 학습형 접촉 예측기). 9분 데이터에서 가장 값싼 보험 → `fusion: gated`.
- **flow matching 을 기본으로 삼지 말 것.** ManiFeel 이 같은 조건에서 DP·Equivariant DP 와
  비교했고 flow 가 둘 다에 뒤졌다. 옵션으로만 둔다.
- **ResNet-18 scratch 가 촉각 foundation model 들과 대등했다**(ManiFeel). 그리고 Paxini 용
  사전학습 인코더는 3DTacDex 것 외에 존재하지 않는다 → 작게 처음부터 학습이 기본값.

### 평가 방법 경고

`물체당 10 trial × 5 task = 50 rollout` 은 **Clopper-Pearson CI 폭이 20~30%p** 라
두 정책을 가를 검정력이 없다(TRI LBM). 성공률 옆에 **milestone rubric** 을 붙여야 한다 —
"미끄러졌으나 회수" 같은 촉각 고유 사건이 성공률엔 안 잡히는데, 그게 정확히 낙하 회복 태스크다.

---

## 학습

```bash
# 0) 데이터에 뭐가 있는지 먼저 판정
python3 tools/inspect_h5.py /home/js/Desktop/vive_franka_teleop/record/logs

# 1) 데이터·shape·ETA 만 (모델 안 돌림)
python3 train.py --config configs/01_anchor.yaml --mode preflight

# 2) 몇 스텝만 돌려 파이프라인 증명
python3 train.py --config configs/01_anchor.yaml --mode smoke

# 3) 본 학습
python3 train.py --config configs/01_anchor.yaml --mode full
python3 train.py --config configs/08_freq_1to3.yaml --override model.fusion=gated
```

- **best 는 노이즈 MSE 가 아니라 홀드아웃 액션 MAE[count] 로 고른다.**
  diffusion/flow/bc 를 같은 자로 재야 비교가 되고, 이게 배포 성능에 더 가깝다.
  `eval_every: 5` · `eval_n: 768` · `eval_seed: 7` (refer/ 와 같은 규약).
  매 에폭 홀드아웃 전체를 재면 eval 이 에폭 시간의 **56%** 를 먹으면서 지표는 사실상 같다.
- 정규화 통계 + `prefer_weights`(raw/ema) 가 **체크포인트 안에** 들어간다 →
  배포 시 전처리 불일치도, "어느 가중치를 쓰나" 도 구조적으로 정해진다.
- `last.pt` 자동 재개 — **LR 스케줄(`lr_sched`)까지 복원**한다. 없으면 warmup 부터 다시 간다.
- 저장은 tmp → `os.replace` (142MB 를 덮어쓰다 죽어도 기존 체크포인트가 안 깨진다).
- SIGTERM 은 현재 epoch 을 마치고 저장한 뒤 종료한다 (`timeout`·`kill`·`scancel` 처럼
  프로세스 **그룹**에 신호가 가서 DataLoader worker 가 먼저 죽는 경우까지 처리).
- 상수 채널은 **std=1** 로 둔다. refer/ 의 `std.clip(1e-6)` 은 train 에서 상수였던 채널이
  배포 때 조금만 움직여도 정규화 값을 10^6 배로 튀긴다.

### 실데이터 없이 검증하기

```bash
python3 tools/make_fake_h5.py --out /tmp/fake --demos 4 --steps 1000
python3 train.py --config configs/01_anchor.yaml --data /tmp/fake --mode smoke
python3 tests/test_data.py
```

`make_fake_h5.py` 는 레코더와 **동일한 형식**(vlen JPEG + `41_rgb_index` 간접참조 + 카메라
attrs)으로 합성 데이터를 만든다. 여기서 통과하면 남는 실패는 진짜 데이터 고유의 문제뿐이다.

---

## 아직 없는 것

- [ ] `run.py` 배포 연결 + RGB decode 지연 재측정 (20Hz 예산 안에 드는지)
- [ ] future-force 예측 auxiliary loss (AdapTac: 50%→90%)
- [ ] CoP 표현 (`beyond-binary` 의 Tikhonov 최소자승) — 1524-D 해독 후
- [ ] 체크포인트에 frozen backbone 이 3벌(policy/ema/optimizer) 들어가 142MB —
      frozen 파라미터 제외하면 크게 준다

---

## 개발

```bash
python -m venv .venv && .venv/bin/pip install torch torchvision numpy h5py pyyaml pillow
python tests/test_shapes.py     # 조합 50개
python tests/test_configs.py    # config 10개 + 검증 4개
python tests/test_data.py       # 로더 11개 (합성 HDF5 자동 생성)
```

CPU 로 돌고 데이터도 GPU 도 필요 없다. **연구실 가기 전에 둘 다 통과해야 한다** —
여기서 걸리는 버그는 전부 실데이터와 무관한 버그라 여기서 잡는 게 압도적으로 싸다.
