# 텐서 계약 (I/O contract)

이 문서가 **단일 진실**이다. 모듈을 갈아끼워도 이 shape 은 변하지 않는다.
새 인코더/fusion/head 를 추가할 때 여기 적힌 인터페이스만 지키면 조합이 자동으로 성립한다.

---

## 0. 기호

| 기호 | 뜻 | 현재 값 |
|---|---|---|
| `B` | batch | 64 |
| `T_k` | 모달리티 `k` 의 관찰 히스토리 길이 | **모달리티마다 다르다** (↓) |
| `T_p` | 예측 액션 길이 (pred horizon) | 16 |
| `T_a` | 실행 액션 길이 (exec horizon) | 8 |
| `A` | action 차원 | 16 |
| `D` | 토큰 차원 (`d_model`) | 256 |
| `C` | pooled 조건 벡터 차원 (`cond_dim`) | 256 |

시간축: 원본 100Hz, `ds_stride=5` → 실효 **20Hz**. 윈도우 내부만 stride 를 걸고
시작점은 100Hz 매 스텝(= dilated window + dense start, refer/dp_data.py 의 설계를 유지).

---

## 1. refer/ 의 현재 계약 (baseline, 측정으로 확인함)

```
obs   (B, 2, 46)  float32   ← T_obs=2, 모든 모달리티가 한 벡터에 concat
act   (B, 16, 16) float32   ← T_p=16, A=16
```

`46 = 03_hand_j_pos(16) + 06_hand_j_kin(12) + 20_paxini_ft(12) + 30_fruit_pos(3) + 32_fruit_size(3)`

내부 흐름:

```
obs (B,2,46) ──flatten(1)──▶ (B,92) ──MLP──▶ global_cond (B,256)
                                                  │
timestep (B,) ──SinusoidalPosEmb─▶ (B,128) ───────┤
                                                  ▼
noisy_act (B,16,16) ──────────────────▶ denoiser ──▶ eps_pred (B,16,16)
```

- denoiser 조건화: U-Net 은 `cond=(B,384)` 를 **FiLM**(ConvResBlock 마다 `Linear(384→2*out_ch)`),
  DiT 는 같은 `cond` 를 **adaLN-Zero**(`Linear(384→6*d_model)`) 로 받는다.
- 파라미터 실측: `dp_unet` 2.47M / `dp_dit` 5.96M / `bc_mlp` 0.62M.

**이 계약의 한계 3가지** — wrapper 가 푸는 문제:
1. 모달리티가 첫 Linear 에서 구분 없이 섞인다 (인코더 분리 불가).
2. `T_obs` 가 전 모달리티 공통이라 100Hz 촉각의 시간 해상도를 못 쓴다.
3. 이미지가 들어갈 자리가 없다.

---

## 2. wrapper 의 계약

### 2.1 batch (dataset → policy)

```python
batch = {
  "obs": {
    "state":   (B, T_state, D_state),          # float32, 정규화됨
    "tactile": (B, T_tac,   D_tac),            # float32, 정규화됨
    "rgb":     (B, T_rgb,   3, H, W),          # float32 [0,1], ImageNet norm 은 인코더 안에서
  },
  "action":    (B, T_p, A),                    # float32, 정규화됨
  "mask": {                                    # 선택 — modality dropout 용
    "rgb":     (B,) bool,                      # False = 이 샘플에서 rgb 를 가린다
    "tactile": (B,) bool,
  },
}
```

`obs` 의 키는 **config 가 결정한다.** 없는 키는 dict 에 아예 없고, 인코더도 안 만들어진다.
→ `only rgb` / `only tactile` / `rgb+tactile` 이 코드 분기 없이 config 만으로 나온다.

**현재 후보 값**

| key | 구성 | 차원 | 기본 `T` |
|---|---|---|---|
| `state` | `03_hand_j_pos(16)` + `06_hand_j_kin(12)` | 28 | 2 |
| | (+ `30_fruit_pos(3)` + `32_fruit_size(3)` 는 옵션) | +6 | |
| `tactile` | `20_paxini_ft(12)` | 12 | **8** |
| | 또는 `+ contact_bits(4)` | 16 | |
| | 또는 `21_paxini_raw(1524)` | 1524 | |
| | 또는 `cop(4×6=24)` | 24 | |
| `rgb` | `40_rgb_jpeg` → decode → resize | 3×224×224 | 2 |
| `action` | `04_hand_j_tar(16)` 절대 관절 타겟 | 16 | `T_p=16` |

⚠️ `rgb` 와 `21_paxini_raw`, `31_fruit_quat` 은 **데이터에 있는지 미확인**이다.
`tools/inspect_h5.py` 로 먼저 판정할 것.

### 2.2 인코더 인터페이스

```python
class ObsEncoder(nn.Module):
    out_dim: int      # 토큰 하나의 차원 (= D)
    n_tokens: int     # 이 인코더가 내놓는 토큰 수
    def forward(self, x: Tensor) -> Tensor:   # (B, T_k, ...) -> (B, n_tokens, D)
```

**항상 토큰 시퀀스 `(B, n_tokens, D)` 를 낸다.** pooled 벡터가 필요하면 fusion 이 pooling 한다.
이 규약 하나로 fusion 이 인코더 종류를 몰라도 된다.

| 인코더 | 입력 | `n_tokens` | 비고 |
|---|---|---|---|
| `state.mlp` | (B,T,D_s) | 1 | flatten → MLP (refer/ 와 동일) |
| `tactile.mlp` | (B,T,D_t) | 1 | flatten → MLP. **naive flatten 기준선으로만 남긴다** |
| `tactile.conv1d` | (B,T,D_t) | T | ManipForce `FTEmbed` 이식 (learnable α residual) |
| `tactile.lstm` | (B,T,D_t) | `n_part`×T or `n_part` | 부위별 LSTM(가중치 공유) + 부위 임베딩 |
| `tactile.transformer` | (B,T,D_t) | `n_part`×T or 1 | (부위×시각) 토큰 self-attn, 부위/시간 임베딩 분리 |
| `tactile.taxel_cnn` | (B,T,D_t) | `n_part` | 부위별 CNN (1524-D 해독 후) |
| `vision.smallcnn` | (B,T,3,H,W) | T | scratch CNN + spatial softmax |
| `vision.resnet18` | (B,T,3,H,W) | T (pool) / T×L (patch) | `L=(H/32)·(W/32)` |
| `vision.dinov2` | (B,T,3,H,W) | T (cls·pool) / T×L (patch) | `L=(H/14)·(W/14)`, LoRA 가능 |

### 2.2.1 촉각 — 부위(손가락) 축을 잃지 않는다 ★

`20_paxini_ft`(12) 를 `nn.Linear(12, 256)` 에 바로 넣으면 4손가락 × 3축이 **첫 층에서
전부 섞인다.** 접촉이 손가락 사이를 옮겨 가는 낙하 회복에서 그 정보가 정확히 필요하므로
`n_part` 로 부위를 쪼개고 부위 임베딩을 붙인다.

```
(B, T, n_part·ch) ──reshape──▶ (B, T, n_part, ch) ──Linear(ch,D)──▶ + 부위emb + 시간emb
```

| | 부위 identity | 시간축 | 부위 간 상호작용 |
|---|---|---|---|
| `lstm` (`n_part=4, return_seq=true`) | 완전 분리 (부위를 배치로 접음) | 유지 (T 토큰) | **없음** — fusion/denoiser 가 담당 |
| `transformer` (`n_part=4, use_cls=false`) | 임베딩으로 유지 | 유지 (T 토큰) | attention 이 담당 |

⚠️ **레이아웃 가정**: `x[..., p*ch + c]` 가 부위 `p` 의 채널 `c` (부위-major).
`20_paxini_ft` = 4손가락 × 3축이면 `n_part: 4`. 실제가 `N × 6` 이면 `n_part = shape/6`.
**틀리면 조용히 엉뚱한 손가락을 학습한다** — `tools/inspect_h5.py` 로 먼저 확인할 것.

### 2.2.2 vision — `frozen` 과 정규화는 짝이다 ★

`norm="auto"` 가 알아서 고른다. **직접 주려면 규칙을 지켜야 한다:**

| | 정규화 | backbone 모드 | 근거 |
|---|---|---|---|
| 파인튜닝 (`frozen: false`) | GN 치환 | train | DP 논문 요구 — BN running stat 이 EMA 와 충돌 |
| 얼림 (`frozen: true`) | **BN 유지** | eval 고정 | stat 이 갱신되지 않아 충돌이 없다. GN 으로 바꾸면 사전학습 통계만 버린다 |

frozen + GN 을 실측하면 이미지-의존 성분이 **72.8% → 15.3%**, 서로 다른 이미지의
feature 코사인이 **0.471 → 0.976** 이 된다 — 시각 분기가 사실상 상수를 낸다.

DINOv2 는 `lora_rank>0` 으로 base 를 얼린 채 qkv/proj 만 학습한다(ViT-S 기준 221k).
**LoRA 를 켜면 backbone 활성값을 전부 들고 있어야 하므로**(frozen+`no_grad` 때는 0)
`grad_ckpt: true` 가 필요하다 — 실측 6.81 → 2.06 GiB, step +30%.

### 2.3 fusion 인터페이스

```python
class Fusion(nn.Module):
    cond_dim: int        # pooled 벡터 차원 (= C)
    n_cond_tokens: int   # 토큰 시퀀스 길이
    def forward(self, feats: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        #  feats: {modality: (B, n_k, D)}
        #  returns: (tokens (B, N, D),  pooled (B, C))
```

**두 형태를 모두 낸다.** head 가 필요한 쪽만 쓴다.

| fusion | 메커니즘 | pooled |
|---|---|---|
| `concat` | 토큰 전부 concat. **S0 baseline** | `pool: flatten`(기본) or `mean` |
| `passthrough` | 파라미터 0. refer/ 기준선 재현 전용 | mean (투영층이 없으니 어쩔 수 없다) |
| `cross_attn` | 양방향 cross-attention (ManipForce 방식) | `pool: flatten`(기본) or `mean` |
| `gated` | 접촉 세기로 촉각 분기 gating | flatten |

⚠️ `unet1d`/`dit` 은 `needs="pooled"` 라 **pooled 만 본다.** pooled 를 `mean` 으로 뭉개면
부위·시간 토큰 구조가 그 자리에서 사라진다 — 부위 토큰을 만들어 놓고 mean 하면 헛일이다.
그래서 `concat`/`cross_attn` 의 기본이 `flatten` 이다. 토큰을 그대로 쓰려면
`denoiser: transformer`(`needs="tokens"`).

### 2.4 action head 인터페이스

```python
class ActionHead(nn.Module):
    needs: str    # "pooled" | "tokens" | "both"
    def compute_loss(self, action: Tensor, cond) -> Tensor:      # scalar
    def sample(self, cond, generator=None) -> Tensor:            # (B, T_p, A)
```

| head | 조건화 | `needs` |
|---|---|---|
| `dp_unet` | FiLM, cond → `Linear(C+128 → 2·ch)` | pooled |
| `dp_dit` | adaLN-Zero, cond → `Linear(C+128 → 6·d)` | pooled |
| `dp_transformer` | cross-attention (액션 토큰이 obs 토큰을 query) | tokens |
| `flow` | 위 셋 중 하나를 backbone 으로, 목적함수만 rectified flow | 동일 |
| `bc` | 확산 없이 회귀 (기준선) | pooled |

diffusion 과 flow 는 **denoiser 를 공유한다.** 바뀌는 건 목적함수와 샘플러뿐:

| | 학습 목표 | 추론 |
|---|---|---|
| DDPM/DDIM | `MSE(eps_pred, eps)`, cosine T=100 | DDIM 10 step |
| flow matching | `MSE(v_pred, eps - x0)`, `τ~Beta(1.5,1)` | forward Euler 4~10 step |

### 2.5 policy 조립

```
obs dict ──▶ 모달리티별 encoder ──▶ {k: (B,n_k,D)} ──▶ fusion ──▶ (tokens, pooled)
                                                                      │
                                                        action head ◀─┘
```

---

## 3. 배포 시 계약 (`run.py` 호환)

배포는 **1 배치**로 돈다. `run.py` 가 이미 지키는 규약을 그대로 유지한다:

```
obs (1, T_k, ...) ──▶ policy.sample() ──▶ (1, T_p, 16) ──denormalize──▶ count
                                        → 앞 T_a 스텝만 실행 (receding horizon)
```

정규화 통계는 **체크포인트 안에** 넣는다 (refer/ 와 동일). 전처리 불일치를 구조적으로 차단.
⚠️ RGB 를 쓰면 배포 루프에 JPEG decode + resize 가 들어간다 —
`bench_latency.py` 로 20Hz 예산 안에 들어오는지 **반드시** 재측정할 것.

### 3.1 체크포인트 계약 (refer/run.py 와 호환)

`train.py:save_ckpt` 가 쓰는 키. 원자적으로 저장한다(tmp → `os.replace`).

| 키 | 왜 필요한가 |
|---|---|
| `policy` · `ema` | 가중치 |
| `optimizer` · **`lr_sched`** | 재개. `lr_sched` 가 없으면 **LR 이 warmup 부터 다시 시작한다** |
| `obs_norm` · `act_norm` | 배포 전처리 |
| `config` | `build_policy(ck["config"])` 로 정책을 그대로 재조립 |
| **`best_metric.prefer_weights`** | raw/ema 중 어느 쪽이 좋았는지. `run.py --weights auto` 가 읽는다 |
| `epoch` · `gstep` · `best_mae` | 진행 상황 |

---

## 4. 불변식 (테스트가 강제한다)

1. 모든 인코더 출력은 `(B, n_tokens, D)`, `D` 는 전 모달리티 공통.
2. 모든 fusion 출력은 `(tokens (B,N,D), pooled (B,C))`.
3. 모든 head 의 `sample()` 은 `(B, T_p, A)`.
4. `compute_loss()` 는 0-dim scalar 이고 backward 가 통해야 한다.
5. 모달리티를 빼도 (`only tactile` 등) 1~4 가 그대로 성립한다.

`tests/test_shapes.py` 가 config 곱집합을 돌며 위를 전부 검사한다.
