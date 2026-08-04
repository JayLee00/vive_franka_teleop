#!/usr/bin/env bash
# 본 학습을 tmux 세션에서 띄우고 파일 로그를 남긴다.
#   ./launch_train.sh [out_dir] [epochs] [추가 인자...]
# 붙기:   tmux attach -t dp_lemon      떼기: Ctrl+b d
# 로그:   <out_dir>/train.log (train.py 자체 Tee) + <out_dir>/tmux.log (세션 전체)
# 중단:   tmux send-keys -t dp_lemon C-c    (이번 에폭 끝에 저장 후 종료)
# 재개:   같은 out_dir 로 다시 실행 → last.pt 에서 auto-resume
set -euo pipefail

SESSION=${SESSION:-dp_lemon}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${1:-runs/dp_lemon_final}
EPOCHS=${2:-1500}
shift || true; shift || true

mkdir -p "$HERE/$OUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "이미 세션 '$SESSION' 이 있습니다. 붙기: tmux attach -t $SESSION"
  echo "새로 시작하려면: tmux kill-session -t $SESSION"
  exit 1
fi

CMD="cd '$HERE' && python3 -u train.py --mode full --out_dir '$OUT' --epochs $EPOCHS $* 2>&1 | tee -a '$HERE/$OUT/tmux.log'"
tmux new-session -d -s "$SESSION" -n train "bash -lc \"$CMD; echo; echo '[tmux] 종료 코드 '\\\$?; exec bash\""
echo "tmux 세션 '$SESSION' 시작"
echo "  붙기 : tmux attach -t $SESSION"
echo "  로그 : $HERE/$OUT/train.log"
echo "         $HERE/$OUT/tmux.log"
echo "  진행 : tail -f $HERE/$OUT/metrics.csv"
