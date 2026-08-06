"""
실험 No2 + RealSense: SHM 1kHz HDF5 로깅과 RealSense 컬러 PNG 저장을 동시에 수행하며,
두 스트림을 time.perf_counter() 기준으로 동기화.

- 로봇 데이터: 기존과 동일 (S/E 마커, demo0, demo1, ..., 19_real_time_global).
  S 수신 시에만 수집, E 수신 시에만 저장. S 재수신 시 해당 구간 버리고 다시 수집.
- 카메라: 연결된 RealSense를 모두(또는 --num-cameras 개수만큼) 사용.
  시리얼 번호 오름차순으로 cam0, cam1, ... 로 매핑(매 실행마다 같은 물리 카메라 = 같은 cam 번호).
  S~E 구간에서만 PNG 저장. 경로: python/PNG/날짜시간/Demo0/cam0, Demo0/cam1, Demo1/cam0, ...
  S 재수신 시 해당 데모의 해당 카메라 폴더만 비우고 다시 저장.
- 동기화: HDF5에는 camera 그룹 없음.
  demoN/20_real_png        = cam0 의 PNG 링크 (PNG/날짜시간/DemoN/cam0/frame_xxx.png)
  demoN/21_real_png_cam1   = cam1 의 PNG 링크 (PNG/날짜시간/DemoN/cam1/frame_xxx.png)
  ... 카메라가 더 있으면 2C_real_png_camC 형식으로 추가.

동기화 검증:
  - demoN/19_real_time_global[i] = 로봇 i번째 샘플 시점, demoN/2X_real_png[i] = 해당 시점의 PNG 파일명(링크).
  - 검증: python verify_sync.py logs/exp2_YYYYMMDD_HHMMSS.h5
"""
import argparse
import queue
import shutil
import signal
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import h5py

from shm_common import ShmAccess, SHM_MSG_KEY, shmmsgs_to_arrays

DT_1K = 0.001
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_FPS = 30

DATASET_NAMES = [
    "01_hand_mode",
    "02_hand_servo_on",
    "03_hand_j_pos",
    "04_hand_j_tar",
    "05_hand_j_cur",
    "06_hand_j_kin",
    "07_hand_j_tac",
    "08_hand_tip_pos",
    "09_hand_tip_quat",
    "10_franka_Arm_j_pos",
    "11_franka_Arm_j_tar",
    "12_franka_Arm_j_vel",
    "13_franka_Arm_C_pos",
    "14_franka_Arm_j_tq",
    "15_franka_Arm_speed_factor",
    "16_glove_g_pos",
    "17_glove_g_tac",
    "18_real_time_demo",
    "19_real_time_global",
    "20_real_png",  # cam0 PNG 링크. cam1 이상은 finally에서 2X_real_png_camX 로 동적 생성.
]

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = _SCRIPT_DIR / "logs"
DEFAULT_PNG_BASE = _SCRIPT_DIR / "PNG"


def png_dataset_name(cam_id: int) -> str:
    """카메라별 PNG 링크 데이터셋 이름. cam0 은 하위호환을 위해 20_real_png."""
    if cam_id == 0:
        return "20_real_png"
    return f"2{cam_id}_real_png_cam{cam_id}"


def discover_realsense_serials(max_cameras: Optional[int] = None) -> List[str]:
    """연결된 RealSense 장치의 시리얼 번호를 오름차순으로 반환. 없으면 빈 리스트."""
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        print(f"[카메라] pyrealsense2 없음: {e} — RealSense 비활성화")
        return []
    try:
        ctx = rs.context()
        serials = [
            dev.get_info(rs.camera_info.serial_number)
            for dev in ctx.query_devices()
        ]
        serials.sort()  # 매 실행마다 같은 물리 카메라가 같은 cam 번호가 되도록 고정
    except Exception as e:
        print(f"[카메라] 장치 검색 실패: {e}")
        return []
    if max_cameras is not None:
        serials = serials[:max_cameras]
    return serials


def keyboard_reader_thread(out_queue: queue.Queue, stop: threading.Event) -> None:
    """키보드 Enter 입력을 토글로 처리: 한 번 누르면 로깅 시작(S), 다시 누르면 종료(E)."""
    print("\n" + "=" * 60)
    print("  [입력] Enter 를 누르면 로깅 시작, 다시 Enter 를 누르면 로깅 종료")
    print("         (전체 종료는 Ctrl+C)")
    print("=" * 60 + "\n")
    recording = False
    while not stop.is_set():
        try:
            line = sys.stdin.readline()
        except Exception:
            break
        if stop.is_set():
            break
        if line == "":  # EOF (예: stdin 닫힘)
            break
        # 어떤 줄이든(빈 줄 = 그냥 Enter 포함) 한 번의 입력으로 간주하고 토글
        recording = not recording
        out_queue.put("S" if recording else "E")


def realsense_capture_thread(
    cam_id: int,
    serial: str,
    stop: threading.Event,
    timestamps_list: list,
    timestamps_lock: threading.Lock,
    state_dict: dict,
    state_lock: threading.Lock,
    png_base_dir_ref: list,
) -> None:
    """한 대의 RealSense 컬러 프레임을 S~E 구간에서만 PNG로 저장.
    경로: python/PNG/날짜시간/Demo_{demo_idx}/cam{cam_id}/frame_xxx.png"""
    try:
        import cv2
        import pyrealsense2 as rs
    except ImportError as e:
        print(f"[카메라{cam_id}] 의존성 없음: {e} — 비활성화")
        return

    try:
        pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)  # 이 스레드는 이 시리얼의 카메라에만 바인딩
        config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FRAME_FPS)
        pipeline.start(config)
        print(f"[카메라{cam_id}] RealSense {serial} {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FRAME_FPS}fps — S~E 구간만 PNG 저장")
    except Exception as e:
        print(f"[카메라{cam_id}] 초기화 실패({serial}): {e} — 이 카메라는 건너뜁니다.")
        return

    frame_index = 0
    local_epoch = -1  # 메인 스레드의 epoch 와 비교해 폴더 리셋 시점을 카메라마다 독립적으로 판단

    def clear_cam_folder(base_dir: Path, demo_idx: int) -> None:
        d = base_dir / f"Demo_{demo_idx}" / f"cam{cam_id}"
        if d.exists():
            for f in d.iterdir():
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    try:
        while not stop.is_set():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=500)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
            except Exception:
                if stop.is_set():
                    break
                continue

            t = time.perf_counter()
            with state_lock:
                recording = state_dict["recording"]
                demo_idx = state_dict["demo_idx"]
                epoch = state_dict["epoch"]

            if not recording:
                continue

            base_dir = png_base_dir_ref[0] if png_base_dir_ref else None
            if base_dir is None:
                continue

            # epoch 은 매 S 마다 증가. 새 데모/같은 데모 재수집(S 재수신) 모두 여기서 처리.
            if epoch != local_epoch:
                clear_cam_folder(base_dir, demo_idx)
                frame_index = 0
                local_epoch = epoch

            img = np.asanyarray(color_frame.get_data())
            cam_dir = base_dir / f"Demo_{demo_idx}" / f"cam{cam_id}"
            cam_dir.mkdir(parents=True, exist_ok=True)
            png_path = cam_dir / f"frame_{frame_index:06d}.png"
            cv2.imwrite(str(png_path), img)

            with timestamps_lock:
                timestamps_list.append((cam_id, demo_idx, frame_index, t))
            frame_index += 1
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


def _write_png_links(f, ts_copy: list, date_time_dir: str) -> None:
    """카메라별로 (데모, 프레임) 타임스탬프를 로봇 19_real_time_global 에 맞춰 정렬, 2X_real_png 데이터셋 작성."""
    by_cam_demo = defaultdict(list)
    for cam_id, d, i, t in ts_copy:
        by_cam_demo[(cam_id, d)].append((i, t))

    for (cam_id, d) in sorted(by_cam_demo.keys()):
        demo_grp_key = f"Demo_{d}"
        if demo_grp_key not in f:
            continue
        robot_grp = f[demo_grp_key]
        if "19_real_time_global" not in robot_grp:
            continue
        robot_ts = robot_grp["19_real_time_global"][:]
        frame_list = sorted(by_cam_demo[(cam_id, d)], key=lambda x: x[0])
        frame_ts = np.array([t for _, t in frame_list], dtype=np.float64)
        frame_indices = np.array([i for i, _ in frame_list], dtype=np.int64)
        if len(frame_ts) == 0:
            continue

        rel = f"{date_time_dir}/Demo_{d}/cam{cam_id}"
        # 1) 로봇 스텝별로 가장 가까운 PNG 할당
        png_names = []
        for t in robot_ts:
            j = int(np.argmin(np.abs(frame_ts - t)))
            png_names.append(f"{rel}/frame_{frame_indices[j]:06d}.png")
        # 2) PNG가 더 느리면: 촬영된 모든 PNG가 최소 한 번은 등장하도록 보정
        for frame_idx in frame_indices:
            needle = f"frame_{frame_idx:06d}.png"
            if not any(needle in s for s in png_names):
                pos = np.where(frame_indices == frame_idx)[0]
                if len(pos) > 0:
                    t_frame = frame_ts[pos[0]]
                    step_idx = int(np.argmin(np.abs(robot_ts - t_frame)))
                    png_names[step_idx] = f"{rel}/frame_{frame_idx:06d}.png"

        robot_grp.create_dataset(
            png_dataset_name(cam_id),
            data=np.array(png_names, dtype=object),
            dtype=h5py.special_dtype(vlen=str),
        )


def run_logger(
    out_path: Path,
    shm_key: int = SHM_MSG_KEY,
    use_camera: bool = True,
    num_cameras: Optional[int] = None,
) -> None:
    shm = ShmAccess(key=shm_key)
    if not shm.attach():
        raise SystemExit("SHM attach 실패 (C++ 프로세스가 먼저 SHM을 생성해야 함).")

    stop = threading.Event()
    event_queue = queue.Queue()

    png_base_dir_ref: list = []  # [Path] set on first S
    camera_timestamps: list = []  # (cam_id, demo_idx, frame_index, t)
    camera_timestamps_lock = threading.Lock()
    # epoch: 매 S 마다 +1. 카메라 스레드들이 폴더 리셋 시점을 잡는 데 사용 (멀티 카메라 안전).
    state_dict = {"recording": False, "demo_idx": 0, "epoch": 0}
    state_lock = threading.Lock()
    camera_threads: List[threading.Thread] = []

    if use_camera:
        serials = discover_realsense_serials(max_cameras=num_cameras)
        if not serials:
            print("[카메라] 연결된 RealSense 없음 — SHM만 로깅합니다.")
        for cam_id, serial in enumerate(serials):
            th = threading.Thread(
                target=realsense_capture_thread,
                args=(cam_id, serial, stop, camera_timestamps, camera_timestamps_lock,
                      state_dict, state_lock, png_base_dir_ref),
                daemon=True,
            )
            th.start()
            camera_threads.append(th)
        if serials:
            print(f"[카메라] {len(serials)}대 대기 (S 수신 시 python/PNG/날짜시간/DemoN/cam0,cam1,... 저장)")

    def on_signal(signum, frame):
        stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    reader = threading.Thread(
        target=keyboard_reader_thread,
        args=(event_queue, stop),
        daemon=True,
    )
    reader.start()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = None
    demo_idx = 0
    current_demo = None
    t_demo_start = 0.0
    buffer = []
    t_start = time.perf_counter()
    sample_count = 0

    try:
        while not stop.is_set():
            t_cycle = time.perf_counter()
            msg = shm.read()
            arrs = shmmsgs_to_arrays(msg)

            if current_demo is not None:
                t_now = time.perf_counter()
                arrs["18_real_time_demo"] = np.array(t_now - t_demo_start, dtype=np.float64)
                arrs["19_real_time_global"] = np.array(t_now, dtype=np.float64)
                buffer.append(arrs)

            try:
                while True:
                    token = event_queue.get_nowait()
                    if token == "S":
                        if f is None:
                            f = h5py.File(out_path, "w")
                            f.attrs["rate_hz"] = 1000.0
                            f.attrs["start_time"] = datetime.now().isoformat()
                            png_base_dir_ref.append(DEFAULT_PNG_BASE / datetime.now().strftime("%Y%m%d_%H%M%S"))
                        if current_demo is True:
                            print(f"[로거] 재시작 → 이전 버퍼 비우고 Demo_{demo_idx} 다시 수집")
                        current_demo = True
                        t_demo_start = time.perf_counter()
                        buffer = []
                        with state_lock:
                            state_dict["recording"] = True
                            state_dict["demo_idx"] = demo_idx
                            state_dict["epoch"] += 1  # 새 데모/재수집 모두 epoch 증가 → 카메라가 폴더 리셋
                        print(f"\n>>> 로깅 시작 (Demo_{demo_idx})")
                    elif token == "E":
                        if buffer:
                            n_steps = len(buffer)
                            grp = f.create_group(f"Demo_{demo_idx}")
                            for name in DATASET_NAMES:
                                if "real_png" in name:
                                    continue  # finally 블록에서 카메라 타임스탬프 기준으로 채움
                                stacked = np.stack([b[name] for b in buffer])
                                grp.create_dataset(
                                    name,
                                    data=stacked,
                                    compression="gzip",
                                )
                            grp.attrs["n_samples"] = n_steps
                            print(f"<<< 로깅 종료 (Demo_{demo_idx}, 총 {n_steps} step 저장)\n")
                            demo_idx += 1
                        else:
                            print(f"<<< 로깅 종료 (Demo_{demo_idx}, 수집된 step 없음 → 건너뜀)\n")
                        buffer = []
                        current_demo = None
                        with state_lock:
                            state_dict["recording"] = False
            except queue.Empty:
                pass

            sample_count += 1
            elapsed = time.perf_counter() - t_cycle
            if elapsed < DT_1K:
                time.sleep(DT_1K - elapsed)
    except Exception as e:
        print(f"로거 오류: {e}")
        print("  → E를 받아 이미 저장한 demo는 파일에 유지됩니다.")
    finally:
        stop.set()
        for th in camera_threads:
            th.join(timeout=2.0)
        shm.detach()

        if f is not None:
            with camera_timestamps_lock:
                ts_copy = list(camera_timestamps)
            if ts_copy and png_base_dir_ref:
                date_time_dir = png_base_dir_ref[0].name  # e.g. 20260314_175005
                _write_png_links(f, ts_copy, date_time_dir)
            f.attrs["sample_count"] = sample_count
            # Ctrl+C 등으로 종료 시 HDF5에 저장된 데모에 해당하는 PNG만 남기고, 나머지 Demo 폴더 삭제
            saved_demos = {int(k[5:]) for k in f.keys() if k.startswith("Demo_") and k[5:].isdigit() and isinstance(f[k], h5py.Group)}
            f.close()
            if png_base_dir_ref:
                base_dir = png_base_dir_ref[0]
                for sub in base_dir.iterdir():
                    if sub.is_dir() and sub.name.startswith("Demo_"):
                        try:
                            idx = int(sub.name[5:])
                            if idx not in saved_demos:
                                shutil.rmtree(sub, ignore_errors=True)  # cam0/cam1 하위 폴더까지 재귀 삭제
                                print(f"[로거] HDF5에 없는 데모 → PNG 폴더 삭제: {sub.name}")
                        except ValueError:
                            pass

        with camera_timestamps_lock:
            ts_copy = list(camera_timestamps)
        if ts_copy and not png_base_dir_ref:
            base = DEFAULT_PNG_BASE / datetime.now().strftime("%Y%m%d_%H%M%S")
            base.mkdir(parents=True, exist_ok=True)
            csv_path = base / "frame_timestamps.csv"
            with open(csv_path, "w") as cf:
                cf.write("cam_id,demo_idx,frame_index,timestamp\n")
                for c, d, i, t in ts_copy:
                    cf.write(f"{c},{d},{i},{t}\n")
            print(f"[카메라] HDF5 미생성 — 타임스탬프만 저장: {csv_path}")

    total_time = time.perf_counter() - t_start
    abs_path = out_path.resolve()
    n_demos = demo_idx if f is not None else 0
    print(f"저장 완료: {abs_path}")
    print(f"  데모 {n_demos}개, 총 샘플 {sample_count}, 실제 {total_time:.2f}s")
    if camera_timestamps and png_base_dir_ref:
        print(f"  카메라 프레임 {len(camera_timestamps)}장, 디렉터리: {png_base_dir_ref[0]}")
    if n_demos == 0 and sample_count > 0:
        print("  [안내] 로깅 종료(2번째 Enter)가 한 번도 없어 demo가 저장되지 않았습니다.")
        print("  → Enter 로 로깅 시작, 다시 Enter 로 로깅 종료해야 해당 구간이 저장됩니다.")


def main():
    parser = argparse.ArgumentParser(
        description="실험 No2 HDF5 로거 + RealSense PNG (타임스탬프 동기화, 멀티 카메라 지원)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="HDF5 출력 경로 (기본: python/logs/exp2_YYYYMMDD_HHMMSS.h5)",
    )
    parser.add_argument(
        "-k", "--shm-key",
        type=lambda x: int(x, 0),
        default=SHM_MSG_KEY,
        help=f"SHM 키 (기본: {hex(SHM_MSG_KEY)})",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="RealSense 비활성화, SHM만 로깅 (기존 exp2와 동일)",
    )
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=None,
        help="사용할 카메라 수 (기본: 연결된 RealSense 전부). 예: 2",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = DEFAULT_LOG_DIR / f"exp2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"

    run_logger(
        Path(args.output),
        args.shm_key,
        use_camera=not args.no_camera,
        num_cameras=args.num_cameras,
    )


if __name__ == "__main__":
    main()
