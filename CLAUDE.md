# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

---

## 0. 수집 데이터는 절대 지우지 않는다 (최우선)

`record/logs/` 와 `hdf5/` 는 **사람이 로봇 앞에서 발판을 밟아가며 모은 원본 데이터**다.
재생성이 불가능하고, 이 저장소의 어떤 코드보다 비싸다.

**금지 — 어떤 이유로도 하지 말 것:**
- `rm record/logs/*.h5`, `rm -rf record/logs`, `rmdir record/logs`
- 그 안의 파일을 이동·이름변경·덮어쓰기 (`mv`, `>`, `h5py.File(..., "w")`)
- "테스트를 깨끗한 상태에서 시작하려고" 비우는 것 ← **실제로 22 데모 9분치를
  이 이유로 날렸다 (2026-08-05, 다른 세션이 `rm -f record/logs/*.h5` 를 6회 실행).
  Trash 미경유라 복구도 불가능했다.**

**레코더/파이프라인을 테스트할 때:**
- 출력은 `/tmp` 또는 `$CLAUDE_JOB_DIR/tmp` 로 보낸다. `--out_dir` 를 쓰거나
  `OUT_DIR` 를 인자로 받게 고친다. 실데이터 폴더를 픽스처로 쓰지 않는다.
- 원본을 읽을 때는 반드시 `h5py.File(path, "r")`. 클램프·전처리는 메모리에서만 한다.

**수집 직후 다른 파티션으로 백업한다:**
```bash
rsync -a ~/Desktop/vive_franka_teleop/record/logs/ /mnt/grasp_data/lemon_logs_backup/
```

지우는 게 꼭 필요하다고 판단되면 **먼저 사용자에게 묻는다.** 예외 없다.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
