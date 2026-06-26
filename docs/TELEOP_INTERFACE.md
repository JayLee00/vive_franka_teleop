# Vive 텔레옵 → Franka 제어 PC 인터페이스 안내 (수신측)

## 네트워크
- ROS2 환경: `ROS_DOMAIN_ID=9`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, `ROS_LOCALHOST_ONLY=0`
- 전용 LAN(직결 GbE): Vive PC `192.168.0.1` ↔ 제어 PC `192.168.0.100`
- WiFi 우회 방지: 첨부 `fastdds_lan_only.xml`(제어PC용, 192.168.0.100 화이트리스트) 적용 후 노드 재시작
  ```bash
  export FASTRTPS_DEFAULT_PROFILES_FILE=~/fastdds_lan_only.xml
  ```

## 구독할 토픽
- `/teleop/delta/right` → **오른팔** (Vive tracker_1)
- `/teleop/delta/left`  → **왼팔**  (Vive tracker_2)
- 타입 `std_msgs/String`(JSON), ~100Hz, RELIABLE
- (참고용) 트래커 절대 포즈: `/vive/{right,left}/pose` (geometry_msgs/PoseStamped)

## 메시지 포맷 (JSON)
```json
{"arm":"right","stamp":1781088016.0,"engaged":true,"valid":true,
 "pos":[dx,dy,dz], "rot":[rx,ry,rz],
 "abs_pos":[x,y,z], "abs_quat":[qx,qy,qz,qw]}
```
| 필드 | 의미 | 단위/형태 |
|---|---|---|
| `pos` | engage 기준 **상대 위치 델타** | 미터 [m] |
| `rot` | engage 기준 **상대 회전 델타** | 라디안 [rad], **rotvec(axis-angle)**: 크기=회전각, 방향=축 |
| `abs_pos` | 트래커 절대 위치 | 미터 [m] (vive_world) |
| `abs_quat` | 트래커 절대 자세 | 단위 쿼터니언 (x,y,z,w) |
| `engaged` | 클러치 on/off | bool |
| `valid` | 트래커 추적 유효 | bool |

좌표계 `vive_world` = SteamVR 월드, **오른손 좌표계, +Y 위**.

## 제어 방법 (권장: 델타)
1. `engaged==false` 또는 `valid==false` → `pos`/`rot`가 0으로 옴 → **현재 자세 유지(hold)**.
2. engage **상승엣지**(false→true) 순간의 현재 EE 포즈 `T_ee0`를 캡처.
3. 매 메시지마다:
   ```python
   from scipy.spatial.transform import Rotation as R
   dR = R.from_rotvec(rot)        # rot: rad rotvec
   dp = pos                       # m
   # T_target = T_ee0 * [dR | dp]
   ```
4. 결과 `T_target`을 EE 목표로 명령.

⚠️ **프레임 정렬(캘리브레이션 포인트)**: `pos`/`rot`은 *트래커(앵커) 로컬 프레임* 기준입니다.
트래커를 잡는 방향과 로봇 EE 방향이 다르면, 고정 정렬회전 `R_align`을 한 번 구해
`dp`,`dR`를 로봇 프레임으로 변환해 적용하세요.

## 검증
- 제어PC에서 `ros2 topic echo /teleop/delta/right` → `engaged:true`에서 트래커 움직이면 `pos`/`rot` 변함.
- 양방향: 제어PC의 `/joint_states` 등이 Vive PC에서 보이면 OK.
