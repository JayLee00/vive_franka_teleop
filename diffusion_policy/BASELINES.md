# 무엇을 쓸까 — 후보 정리와 우리 데이터에 대한 판단

> 결론부터: **지금 병목은 모델이 아니라 (1) 관측이 태스크를 결정하지 못한다는 점과
> (2) 데이터 9분**이다. 아키텍처 교체로 얻을 이득은 이 둘을 고치는 것보다 훨씬 작다.
> 그래서 "DP 를 먼저 학습해 두고" 라는 지시대로 DP(U-Net)를 기본 배포 대상으로 두고,
> 대안들은 같은 홀드아웃에서 실측 비교했다 (아래 표, `runs/sweep/sweep_results.json`).

---

## 0. 우리 상황 요약

| 항목 | 값 |
|---|---|
| 데모 | 22개 (사용 54,183 step @100Hz = **9.0분**) |
| obs | 46차원 저차원 (관절 16 + kin 12 + 촉각 12 + 과일 위치 3 + 과일 크기 3). **이미지 없음** |
| action | 16차원 절대 관절 타겟 [count], 20Hz |
| 태스크 | 레몬 in-hand 회전 + 낙하 회복 |

**구조적 한계**: obs 에 과일의 **각도**가 없다. in-hand 회전은 "지금 몇 도 돌았나"가
액션을 결정하는데 그 변수가 관측에 없으므로 태스크가 부분관측(POMDP)이다. 같은 obs 에
서로 다른 정답 액션이 붙으니 어떤 모델이든 조건부 분포가 다봉(multimodal)이 된다.
→ 확산/CVAE 처럼 **다봉 분포를 표현할 수 있는 정책**이 이론적으로 유리한 상황이고,
동시에 **어떤 모델도 위상을 알 수 없어** 상한이 낮다. (과일 크기 `a`가 카메라에 대한
장축 기울기를 일부 반영하므로 위상 정보가 완전히 0은 아니다.)

---

## 1. 후보군

### A. Diffusion Policy — CNN(1D U-Net) 조건화 · **채택(기본)**
Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023.
- 액션 청크를 확산으로 생성. 다봉 분포·고차원 액션에 강하고, 논문에서 CNN 버전이
  하이퍼파라미터에 **덜 민감**하다고 보고 → 첫 시도에 적합.
- 참조 구현(`R_Franka_KISTAR_Hand/diffusion`)이 이미 이 구조라 재현·비교가 쉽다.
- 단점: 추론에 반복 디노이징 필요(DDIM 10스텝). 우리 측정 latency 는 아래 참조.

### B. Diffusion Policy — Transformer(DiT) 디노이저 · **구현·비교함**
Peebles & Xie, *Scalable Diffusion Models with Transformers*, ICCV 2023 (adaLN-Zero).
- 원 논문(DP)에서도 transformer 버전이 **액션이 빠르게 변하는 태스크**에 유리하다고
  보고. 손가락 게이팅은 실제로 빠르게 변하므로 시도 가치가 있었다.
- 단점: 데이터가 적을 때 과적합이 더 쉽고 LR/warmup 에 민감.

### C. ACT (Action Chunking Transformer + CVAE) · **미구현, 다음 후보 1순위**
Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, RSS 2023.
- 액션 청킹 + **temporal ensembling** + CVAE 로 다봉성 처리. 확산보다 추론이 **1 forward**
  로 끝나 지연이 작고, 정밀 조작(젓가락·지퍼 등)에서 강함.
- 우리에게 매력적인 이유: 16-DoF 손, 50 데모 이하 소량 데이터에서 검증된 사례가 많다.
- temporal ensembling 자체는 **모델과 무관**하므로 이미 `run.py --temporal_ensemble`
  로 우리 DP 에 붙여 놨다(효과는 표에서 측정).

### D. BC-RNN / LSTM-GMM (robomimic) · **소량·저차원에서 강한 기준선**
Mandlekar et al., *What Matters in Learning from Offline Human Demonstrations*, CoRL 2021.
- robomimic 의 큰 교훈: **저차원 관측 + 사람 시연 소량**에서는 GMM 헤드를 단 BC-RNN 이
  훨씬 무거운 방법과 대등하거나 낫다. 우리 조건(46차원, 22데모)에 정확히 해당.
- 우리가 넣은 `bc_mlp` 는 이것의 축소판(순수 MLP·단봉)이다. 표에서 BC 가 DP 를
  이긴다면 다음 수는 **DiT 가 아니라 GMM 헤드 + RNN** 이다.

### E. Flow matching 정책 (π0 계열) · **지금은 과함**
Black et al., *π0: A Vision-Language-Action Flow Model*, 2024.
- rectified flow 로 1~4스텝 추론. 지연이 문제일 때 유효하지만 우리 지연은 이미 충분.

### F. Consistency Policy / one-step 증류 · **지금은 불필요**
Prasad et al., *Consistency Policy*, RSS 2024. DDIM 10스텝이 이미 실시간이라 보류.

### G. VLA (OpenVLA, RDT-1B, Octo) · **해당 없음**
이미지·언어 대규모 사전학습 전제. 우리는 이미지를 obs 로 쓰지 않고 데이터가 9분이다.

### H. 데이터 쪽 기법 · **실제로 가장 큰 지렛대**
- **DAgger / HG-DAgger** (Ross+ 2011, Kelly+ 2019): 1차 정책을 굴리다 실패하기 시작하는
  상태에서 글러브로 개입해 교정을 기록. BC 의 분포 이동을 정면으로 고친다.
  22 데모를 40개로 늘리는 것보다 **효율이 훨씬 높다.**
- **관측 노이즈 증강**: 저차원 BC 의 표준 정규화. 우리 스윕의 핵심 변수.
- **과일 각도 관측 추가**: IMU 내장 또는 오프라인 각도 라벨 → 태스크가 처음으로
  관측 가능해진다. 이게 진짜 해법이다.

---

## 2. 실측 비교 (동일 홀드아웃 3 데모, open-loop MAE)

측정 방식은 `eval_rollout.py` 참조. 배포와 같은 receding-horizon 으로 굴리고,
관측은 GT 궤적에서 읽는다(시뮬레이터 없음) → **액션 예측 정확도**이며 태스크
성공률이 아니다. 숫자는 `runs/sweep/sweep_results.json`.

<!-- SWEEP_TABLE -->

---

## 3. 권고 순서

1. **지금**: DP(U-Net) + 스윕 1위 정규화 설정으로 배포 → `run.py` 로 실기 테스트.
   실기에서 볼 것은 MAE 가 아니라 "게이팅이 주기적으로 나오는가 / 레몬이 흐를 때
   파지를 조이는가" 두 가지.
2. **데이터가 늘기 전에 모델을 바꾸지 말 것.** 22 데모에서 아키텍처 순위는 시드에
   흔들린다(홀드아웃 3개뿐). 표의 차이가 10% 안쪽이면 유의미하지 않다고 봐야 한다.
3. **다음 데이터**: DAgger 방식 교정 데이터 + 과일 각도 관측(IMU). 이 둘이 들어오면
   ACT 또는 BC-RNN(GMM) 을 같은 하니스로 붙여 재비교 (`--algo` 만 추가하면 됨).
4. 양방향 회전을 넣을 땐 **방향 플래그를 obs 에 반드시 추가**. 안 넣으면 같은 관측에
   정반대 액션이 붙어 정책이 평균을 낸다 — 확산이라도 이건 못 고친다(조건이 같으니까).
