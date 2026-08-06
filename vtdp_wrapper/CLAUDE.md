# CLAUDE.md — 이 저장소에서 작업할 때

KISTAR 16관절 손 + Franka, **레몬 in-hand 회전 + 낙하 회복** 태스크의 visuo-tactile
diffusion policy. **논문이 아니라 과제 데모용이다 — "되는 것"이 "새로운 것"보다 우선한다.**

상세 인수인계: [`docs/HANDOFF.md`](docs/HANDOFF.md) · 텐서 계약: [`docs/SHAPES.md`](docs/SHAPES.md)

---

## 먼저 알아야 할 것

- 작성 시점(2026-08-06)까지 **실제 데이터로 학습한 적이 없다.** 전부 합성 HDF5 위에서만
  검증됐다. 실데이터에서 처음 도는 순간이 아직 안 왔다.
- `refer/` 는 **읽기 전용 참고 구현**이다. 고치지 말 것. 여기 있는 `BASELINES.md`/
  `RESULTS.md` 의 실측 수치가 이 프로젝트의 사실 근거다.
- 커밋 전에 테스트 3개가 다 통과해야 한다:
  ```bash
  python tests/test_shapes.py && python tests/test_configs.py && python tests/test_data.py
  ```
  전부 CPU·데이터 없이 돈다(`test_data.py` 는 합성 HDF5 를 스스로 만든다).

## 손대기 전에 확인할 미확인 사항

1. **실데이터에 RGB 가 있는가.** 레코더의 `--rgb-on` 은 opt-in 이라 없을 수 있다.
2. **`21_paxini_raw` 1524-D 가 어떤 구조인가.** 단순 격자로 안 떨어진다.

둘 다 `python3 tools/inspect_h5.py <로그폴더>` 한 번으로 판정된다. **이걸 먼저 돌려라.**
결과에 따라 쓸 수 있는 config 가 갈린다.

## 절대 방향 (이걸 잃으면 프로젝트가 아니다)

1. **촉각의 부위축과 시간축을 뭉개지 않는다.** 손가락마다 나오는 F/T 를 첫 Linear 에서
   섞으면 안 된다 → `n_part` 로 쪼개고 부위 임베딩을 붙인다(`docs/SHAPES.md` 2.2.1).
   `return_seq: true` / `use_cls: false` 로 시간축도 남긴다.
   그리고 pooled 를 `mean` 으로 뭉개면 그 구조가 그 자리에서 사라진다 → `pool: flatten`.
2. **시각은 사전학습을 실제로 쓴다.** resnet18 파인튜닝(backbone LR ×0.1) 또는
   DINOv2 + LoRA. **frozen + BN→GN 치환 조합은 금지** — feature 가 거의 상수가 된다.
3. **action decoder 는 DP(unet1d) × DiT × flow 가 곱집합으로 돌아야 한다.**
   `needs` 계약(pooled/tokens)을 깨지 말 것.

## 깨면 안 되는 불변식

- **인코더 출력은 항상 `(B, n_tokens, d_model)`.** 예외 없음. 이 규약이 조합을 성립시킨다.
- **fusion 출력은 항상 `(tokens, pooled)` 둘 다.** head 가 필요한 쪽만 쓴다.
- **`10_baseline_refer` = refer/dp_unet 파라미터 2,469,200 정확히 일치.**
  `tests/test_configs.py` 가 이제 강제한다.
- **`best` 는 홀드아웃 액션 MAE[count] 로 고른다.** 노이즈 MSE 로 바꾸지 말 것 —
  diffusion/flow/bc 를 같은 자로 못 재게 된다. 샘플링 노이즈는 `eval_seed` 로 고정한다.
- **체크포인트에 `lr_sched` 와 `best_metric.prefer_weights` 가 들어간다.**
  앞은 재개 시 LR 이 warmup 부터 다시 시작하는 걸 막고, 뒤는 `run.py --weights auto` 가 읽는다.
  저장은 tmp → `os.replace` 로 원자적으로.
- **정규화 통계는 train split 프레임만으로.** 그리고 상수 채널은 `std=1`(1e-6 clip 금지 —
  배포 때 10^6 배 폭발한다).
- **RGB 관측은 과거에서만 가져온다.** `41_rgb_index == -1` 구간(카메라 늦게 켜짐)은
  윈도우 생성 단계에서 제외한다. 미래 프레임을 당겨오면 누출이다.

## config 규칙 (직관과 다른 부분)

`_base_` 체인 상속이고 병합 규칙에 **예외 2개**가 있다:

- **`obs_spec` 은 통째 교체** — merge 하면 자식이 부모의 모달리티를 *뺄 수 없다*
- **`*_kwargs` 도 통째 교체** — 모듈을 바꿨는데 부모 kwargs 가 남으면 생성자가 터진다

새 모듈은 `@register(kind, name)` 데코레이터로 등록하고 `docs/SHAPES.md` 의 인터페이스를
지키면 config 에서 바로 쓸 수 있다. Hydra 를 쓰지 않는다(오류 메시지 때문에 의도적).

**`--override` 로 모듈을 바꿀 때는 그 kwargs 도 같이 비워야 한다** (통째 교체 규칙의 결과다):

```bash
python3 train.py --config configs/01_anchor.yaml \
  --override model.denoiser=dit model.denoiser_kwargs={}          # kwargs 를 안 비우면 거부됨
  --override model.fusion=gated model.fusion_kwargs='{gate_on: tactile, floor: 0.1}'
```
`model.fusion_kwargs.gate_on=...` 처럼 **없는 하위 키는 만들 수 없다** — dict 통째로 준다.

## 실험 설계

`configs/01_anchor.yaml` 에서 **한 번에 한 축만** 바꾼 것이 `02`~`08` 이다.
홀드아웃이 3 데모뿐이라 두 축을 동시에 바꾸면 원인을 못 짚는다.
`refer/BASELINES.md` 가 **"10% 안쪽 차이는 유의미하지 않다"** 고 못박아 뒀다.

## 하지 말 것

- flow matching 을 기본으로 삼기 (ManiFeel 이 같은 조건에서 diffusion 에 뒤진다고 보고)
- 1524-D 를 임의 격자로 reshape 해서 gel 이미지 사전학습 인코더에 넣기 (Sparsh·T3·UniT·
  AnyTouch 는 전부 광학 gel 전용이라 압력 배열에 전이되지 않는다)
- open-loop MAE 만 보고 `exec_horizon` 정하기 (`refer/RESULTS.md` 5절: 실기에서 정반대였다)
- 22 데모에 큰 backbone 파인튜닝 (frozen 으로 시작하고, 과소적합이면 그때 풀기)
