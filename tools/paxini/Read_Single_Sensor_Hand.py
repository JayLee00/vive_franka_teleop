"""
실제 UART/serial 통신 담당: 
serial port open, auto-push mode 시작/종료 명령 전송, 센서에서 raw frame 읽기, sample 단위로 frame을 yield, “센서와 말하는 저수준 통신 코드”
"""
import serial
import time
import serial.tools.list_ports
from typing import List, Optional, Dict
import logging

# 장치 프로토콜 설정(GEN3 센서와 일치)
AUTO_PUSH_REG = 0x0017               # 자동 전송 제어 레지스터(1=켜기, 0=끄기)
AUTO_PUSH_FRAME_HEAD = b"\xAA\x56"   # 자동 전송 데이터 프레임 헤더
VERSION_REG = 0x0000                 # 버전 번호 레지스터 주소
VERSION_DATA_LEN = 0x000F            # 버전 번호 데이터 길이(15바이트)
DATA_TYPE_REG = 0x0016               # 데이터 타입 조합 레지스터
BAUDRATE = 921600                    # 고속 통신 baudrate
TIMEOUT_CMD = 1.0                    # 명령 응답 타임아웃(초)
TIMEOUT_AUTO_PUSH = 0.05             # 자동 전송 모니터링 타임아웃(초)

# 프레임 구조 상수
REQ_HEAD = b"\x55\xAA"               # 요청 프레임 헤더(호스트 -> 센서)
RESP_HEAD_GENERAL = b"\xAA\x55"      # 일반 응답 프레임 헤더(센서 -> 호스트)
RESP_HEAD_AUTO_PUSH = b"\xAA\x56"    # 자동 전송 관련 응답 프레임 헤더
RESERVED = b"\x00"                   # 예약 필드
FUNC_READ = 0x03                     # 레지스터 읽기 기능 코드
FUNC_WRITE = 0x10                    # 레지스터 쓰기 기능 코드

# 명령 예시(검사용)
CMD_EXAMPLE_AUTO_PUSH = "55AA00101700010001D8"  # 자동 전송 활성화 예시 명령
CMD_EXAMPLE_VERSION = "55AA000300000F00EF"      # 버전 번호 읽기 예시 명령

# 보정 명령(Hand_UI.py의 '보정 시작'과 동일: func 0x17, reg 0x0002, data 0x01)
DEFAULT_CALIB_CMD_HEX = "55AA00170200010001E6"

# 로그 설정
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sensor_comm.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def send_hex_cmd(ser: serial.Serial, hex_cmd: str) -> bool:
    """16진수 명령을 전송합니다."""
    try:
        cmd_bytes = bytes.fromhex(hex_cmd)
        ser.flushOutput()
        sent_len = ser.write(cmd_bytes)
        if sent_len != len(cmd_bytes):
            logger.error(f"명령 전송이 완전하지 않음: 보내야 할 길이{len(cmd_bytes)}바이트, 실제 전송 길이{sent_len}바이트 | 명령: {hex_cmd}")
            return False
        logger.debug(f"명령 전송 성공: {hex_cmd}({sent_len}바이트)")
        return True
    except (serial.SerialException, ValueError, Exception) as e:
        logger.error(f"전송 오류: {str(e)} | 명령: {hex_cmd}", exc_info=True)
        return False


def read_serial_data(ser: serial.Serial, timeout: float = TIMEOUT_CMD, expected_head: Optional[bytes] = None) -> Optional[bytes]:
    """시리얼 데이터를 읽습니다(지정한 예상 프레임 헤더 지원)."""
    try:
        start_time = time.time()
        recv_data = b""
        while time.time() - start_time < timeout:
            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting)
                recv_data += chunk
                logger.debug(f"데이터 조각 수신: {chunk.hex()}({len(chunk)}바이트)")
                
                # 예상 프레임 헤더가 지정된 경우 수신 여부 확인
                if expected_head and expected_head in recv_data:
                    # 프레임 헤더부터 데이터 자르기
                    head_pos = recv_data.find(expected_head)
                    recv_data = recv_data[head_pos:]
                    break  # 예상 프레임 헤더를 찾으면 대기 종료
                    
                start_time = time.time()  # 타임아웃 타이머 초기화
                time.sleep(0.005)  # 이어질 수 있는 데이터를 기다림
            time.sleep(0.001)
        
        if recv_data:
            logger.debug(f"전체 데이터 수신: {recv_data.hex()}({len(recv_data)}바이트)")
            return recv_data
        logger.warning(f"타임아웃 동안 데이터를 받지 못함({timeout}초)")
        return None
    except Exception as e:
        logger.error(f"읽기 오류: {str(e)}", exc_info=True)
        return None


def calc_lrc(data: bytes) -> int:
    """LRC 체크섬을 계산합니다(누적합 -> 반전 -> 1 더하기 -> 하위 8비트)."""
    try:
        lrc_sum = 0
        for byte in data:
            lrc_sum = (lrc_sum + byte) & 0xFF  # 8비트 누적으로 오버플로 방지
        lrc = ((~lrc_sum) + 1) & 0xFF         # 보수 계산
        logger.debug(f"LRC 계산: {data.hex()} -> 0x{lrc:02X}")
        return lrc
    except Exception as e:
        logger.error(f"LRC 계산 오류: {str(e)}", exc_info=True)
        return 0


def build_request_frame(func_code: int, reg_addr: int, data_len: int, write_data: bytes = b"") -> Optional[str]:
    """요청 프레임을 구성합니다(프로토콜: Head+예약+기능 코드+레지스터 주소+데이터 길이+데이터+LRC)."""
    try:
        # 레지스터 주소와 데이터 길이는 모두 리틀 엔디언 사용(프로토콜 요구사항과 일치)
        reg_addr_bytes = reg_addr.to_bytes(2, byteorder="little")
        data_len_bytes = data_len.to_bytes(2, byteorder="little")
        
        # 프레임 본문 조립(LRC 제외)
        frame_without_lrc = (
            REQ_HEAD + 
            RESERVED + 
            func_code.to_bytes(1, "big") + 
            reg_addr_bytes + 
            data_len_bytes + 
            write_data
        )
        
        # LRC를 계산해 프레임 완성
        lrc = calc_lrc(frame_without_lrc).to_bytes(1, "big")
        full_frame = frame_without_lrc + lrc
        frame_hex = full_frame.hex().upper()
        
        # 예시 명령과 일치하는지 확인
        if func_code == FUNC_WRITE and reg_addr == AUTO_PUSH_REG and write_data == b"\x01" and data_len == 1:
            if frame_hex != CMD_EXAMPLE_AUTO_PUSH:
                logger.warning(f"자동 전송 활성화 명령이 예시와 다름: 생성값{frame_hex}, 예시{CMD_EXAMPLE_AUTO_PUSH}")
        if func_code == FUNC_READ and reg_addr == VERSION_REG and data_len == VERSION_DATA_LEN:
            if frame_hex != CMD_EXAMPLE_VERSION:
                logger.warning(f"버전 번호 읽기 명령이 예시와 다름: 생성값{frame_hex}, 예시{CMD_EXAMPLE_VERSION}")
        
        logger.debug(f"요청 프레임 구성: {frame_hex}({len(full_frame)}바이트)")
        return frame_hex
    except Exception as e:
        logger.error(f"요청 프레임 구성 오류: {str(e)}", exc_info=True)
        return None


def parse_auto_response(response: bytes) -> Optional[Dict]:
    """자동 전송 관련 응답을 파싱합니다(비활성화 명령 응답 포함, 프레임 헤더 AA 56)."""
    try:
        # 기본 검사: 최소 프레임 길이
        if len(response) < 7:
            logger.error(f"자동 전송응답 프레임이 너무 짧음: {len(response)}바이트 | 데이터: {response.hex()}")
            return None
        
        # 프레임 헤더 검사(AA 56)
        if response[:2] != RESP_HEAD_AUTO_PUSH:
            logger.error(f"자동 전송응답 프레임 헤더 오류: {response[:2].hex()} 예상{RESP_HEAD_AUTO_PUSH.hex()}")
            return None
        
        # 기본 필드 추출
        parsed = {
            "head": response[:2].hex(),
            "reserved": response[2],
            "valid_frame_len": int.from_bytes(response[3:5], "little"),  # 유효 프레임 길이: 리틀 엔디언
            "error_code": response[5],
            "valid_data": b"",
            "valid_data_len": 0,
            "lrc_valid": False,
            "lrc_calc": 0,
            "lrc_recv": response[-1] if len(response) >= 7 else 0
        }
        
        # 유효 데이터 길이 계산
        parsed["valid_data_len"] = parsed["valid_frame_len"] - 1
        
        # 유효 데이터 추출
        if parsed["valid_data_len"] > 0:
            data_end_pos = 6 + parsed["valid_data_len"]
            if data_end_pos <= len(response) - 1:  # LRC 위치 예약
                parsed["valid_data"] = response[6:data_end_pos]
            else:
                parsed["valid_data"] = response[6:-1]  # LRC 앞까지 잘라냄
                logger.warning(f"자동 전송응답데이터가 불완전함: 예상{parsed['valid_data_len']}바이트, 실제{len(parsed['valid_data'])}바이트")
        
        # LRC 검사
        if len(response) >= 6 + parsed["valid_data_len"] + 1:
            parsed["lrc_calc"] = calc_lrc(response[:-1])
            parsed["lrc_valid"] = (parsed["lrc_calc"] == parsed["lrc_recv"])
            if not parsed["lrc_valid"]:
                logger.warning(f"자동 전송응답LRC검사 실패: 계산값0x{parsed['lrc_calc']:02X}, 실제0x{parsed['lrc_recv']:02X}")
        
        logger.debug(f"자동 전송응답 파싱 완료: {parsed}")
        return parsed
    except Exception as e:
        logger.error(f"자동 전송응답 파싱 오류: {str(e)} | 데이터: {response.hex()}", exc_info=True)
        return None


def parse_response(response: bytes) -> Optional[Dict]:
    """일반 장치 응답 프레임을 파싱합니다(Head+예약+기능 코드+주소+데이터 길이+데이터+LRC)."""
    try:
        # 기본 검사: 최소 프레임 길이
        if len(response) < 8:  # 데이터가 없을 때 최소 8바이트
            logger.error(f"응답 프레임이 너무 짧음: {len(response)}바이트 | 데이터: {response.hex()}")
            return None
        
        # 프레임 헤더 검사(일반 응답은 AA 55)
        if response[:2] != RESP_HEAD_GENERAL:
            logger.error(f"프레임 헤더 불일치: {response[:2].hex()} 예상{RESP_HEAD_GENERAL.hex()} | 데이터: {response.hex()}")
            return None
        
        # 필드 추출(모든 다중 바이트 필드는 리틀 엔디언)
        parsed = {
            "is_error": False,
            "reserved": response[2],
            "func_code": response[3],
            "reg_addr": int.from_bytes(response[4:6], "little"),
            "data_len": int.from_bytes(response[6:8], "little"),
            "actual_data_len": len(response) - 9 if len(response) > 8 else 0,
            "data": b"",
            "lrc_valid": False,
            "lrc_calc": 0,
            "lrc_recv": response[-1] if len(response) >= 9 else 0
        }
        
        # 오류 응답 처리(기능 코드 최상위 비트가 1이면 오류)
        if (parsed["func_code"] & 0x80) != 0:
            parsed["is_error"] = True
            parsed["error_code"] = parsed["func_code"] & 0x7F
            logger.warning(f"장치 오류: 0x{parsed['error_code']:02X} | 주소0x{parsed['reg_addr']:04X}")
            return parsed
        
        # 기능 코드 검사
        if parsed["func_code"] not in [FUNC_READ, FUNC_WRITE]:
            logger.error(f"유효하지 않은 기능 코드: 0x{parsed['func_code']:02X}")
            return None
        
        # 유효 데이터 추출
        if parsed["data_len"] > 0 and len(response) >= 8 + parsed["data_len"] + 1:
            parsed["data"] = response[8:8 + parsed["data_len"]]
            logger.debug(f"데이터 추출: {parsed['data'].hex()}({len(parsed['data'])}바이트)")
        elif parsed["data_len"] > 0:
            parsed["data"] = response[8:-1] if len(response) > 8 else b""
            logger.warning(f"데이터가 불완전함: 예상{parsed['data_len']}바이트, 실제{len(parsed['data'])}바이트")
        
        # LRC 검사
        if len(response) >= 9:
            parsed["lrc_calc"] = calc_lrc(response[:-1])
            parsed["lrc_valid"] = (parsed["lrc_calc"] == parsed["lrc_recv"])
            if not parsed["lrc_valid"]:
                logger.warning(f"LRC검사 실패: 계산값0x{parsed['lrc_calc']:02X}, 실제0x{parsed['lrc_recv']:02X}")
        
        # 데이터 길이 일치 검사
        if parsed["data_len"] != len(parsed["data"]):
            logger.warning(f"데이터 길이 불일치: 예상{parsed['data_len']}, 실제{len(parsed['data'])}")
        
        return parsed
    except Exception as e:
        logger.error(f"응답 파싱 오류: {str(e)} | 데이터: {response.hex()}", exc_info=True)
        return None


def read_register(ser: serial.Serial, reg_addr: int, read_len: int) -> Optional[bytes]:
    """레지스터 데이터를 읽습니다(기능 코드 0x03)."""
    if not (1 <= read_len <= 512):
        logger.error(f"유효하지 않은 읽기 길이: {read_len}바이트(프로토콜 제한1-512바이트)")
        return None
    
    read_frame = build_request_frame(FUNC_READ, reg_addr, read_len)
    if not read_frame:
        logger.error(f"읽기 요청 구성 실패: 0x{reg_addr:04X}, {read_len}바이트")
        return None
    
    if not send_hex_cmd(ser, read_frame):
        logger.error(f"읽기 요청 전송 실패")
        return None
    
    time.sleep(0.2)  # 응답 시간을 확보
    response = read_serial_data(ser, TIMEOUT_CMD, RESP_HEAD_GENERAL)
    if not response:
        logger.error(f"읽기 응답을 받지 못했습니다")
        return None
    
    parsed = parse_response(response)
    if not parsed or parsed["is_error"] or parsed["func_code"] != FUNC_READ:
        logger.error(f"읽기 작업 실패")
        return None
    
    return parsed["data"]


def write_register(ser: serial.Serial, reg_addr: int, write_data: bytes, is_auto_push: bool = False) -> bool:
    """레지스터 데이터를 씁니다(기능 코드 0x10). is_auto_push는 자동 전송 관련 작업 여부를 나타냅니다."""
    write_len = len(write_data)
    if not (1 <= write_len <= 10):
        logger.error(f"유효하지 않은 쓰기 길이: {write_len}바이트(프로토콜 제한1-10바이트)")
        return False
    
    write_frame = build_request_frame(FUNC_WRITE, reg_addr, write_len, write_data)
    if not write_frame:
        logger.error(f"쓰기 요청 구성 실패: 0x{reg_addr:04X}")
        return False

    # 직전 단계(예: 보정, func 0x17)가 남긴 잔여 AA56 프레임이 이 명령의 응답으로
    # 오인되지 않도록, 명령 전송 직전에 입력 버퍼를 비운다(send_calibration 과 동일 패턴).
    try:
        ser.reset_input_buffer()
    except Exception:
        pass

    if not send_hex_cmd(ser, write_frame):
        logger.error(f"쓰기 요청 전송 실패")
        return False
    
    time.sleep(0.2)  # 응답 시간을 확보
    
    # 자동 전송 관련 작업 여부에 따라 다른 프레임 헤더 선택
    expected_head = RESP_HEAD_AUTO_PUSH if is_auto_push else RESP_HEAD_GENERAL
    response = read_serial_data(ser, TIMEOUT_CMD, expected_head)
    if not response:
        logger.error(f"쓰기 응답을 받지 못했습니다")
        return False
    
    # 응답 파싱
    if is_auto_push:
        parsed = parse_auto_response(response)
        # 자동 전송 응답은 error_code로 성공 여부 판단(0이면 성공)
        if not parsed or parsed["error_code"] != 0:
            logger.error(f"자동 전송관련 쓰기 작업 실패, 오류 코드: 0x{parsed['error_code']:02X}" if parsed else "자동 전송관련 쓰기 작업 응답 파싱 실패")
            return False
    else:
        parsed = parse_response(response)
        if not parsed or parsed["is_error"] or parsed["func_code"] != FUNC_WRITE:
            logger.error(f"쓰기 작업 실패")
            return False
        
        # 쓰기 상태 확인(반환 데이터가 0이면 성공)
        if len(parsed["data"]) > 0:
            write_status = int.from_bytes(parsed["data"], "little")
            if write_status != 0:
                logger.error(f"쓰기 상태 오류: 0x{write_status:02X}(0이면 성공)")
                return False
    
    return True


def disable_auto_push(ser: serial.Serial) -> bool:
    """자동 전송 기능을 끕니다(0x0017 레지스터에 0x00 쓰기)."""
    try:
        # 비활성화 명령을 직접 구성해 전송하고 응답은 기다리지 않음
        disable_cmd = build_request_frame(FUNC_WRITE, AUTO_PUSH_REG, 1, b"\x00")
        if not disable_cmd:
            logger.error("자동 전송 비활성화 명령 구성 실패")
            return False
            
        # 명령을 전송하지만 응답은 검증하지 않음
        if send_hex_cmd(ser, disable_cmd):
            logger.info("자동 전송 비활성화 명령 전송 완료")
            # 명령 수신을 보장하기 위한 짧은 지연
            time.sleep(0.1)
            return True
        return False
    except Exception as e:
        logger.error(f"자동 전송 비활성화 실패: {str(e)}", exc_info=True)
        return False


def enable_auto_push(ser: serial.Serial) -> bool:
    """자동 전송 기능을 켭니다(0x0017 레지스터에 0x01 쓰기)."""
    return write_register(ser, AUTO_PUSH_REG, b"\x01", is_auto_push=True)


def send_calibration(ser: serial.Serial, calib_cmd_hex: str = DEFAULT_CALIB_CMD_HEX) -> bool:
    """센서 보드에 일회성 보정 명령을 전송하고 응답을 검증합니다.

    Hand_UI.py의 '보정 시작' 버튼과 동일한 프레임을 사용합니다. auto-push를
    켜기 *전에* 호출해야 응답(AA55 일반 프레임)을 깨끗하게 읽을 수 있습니다.
    반환값은 보정 성공 여부이며, 실패해도 호출 측에서 스트리밍을 계속할 수
    있도록 예외를 던지지 않습니다.
    """
    calib_cmd_hex = (calib_cmd_hex or DEFAULT_CALIB_CMD_HEX).replace(" ", "").strip()
    try:
        bytes.fromhex(calib_cmd_hex)
    except ValueError:
        logger.error(f"보정 명령 형식 오류(16진수 아님): {calib_cmd_hex}")
        return False

    # 보정 전 입력 버퍼를 비워 이전 잔여 데이터가 응답에 섞이지 않게 함
    try:
        ser.reset_input_buffer()
    except Exception:
        pass

    if not send_hex_cmd(ser, calib_cmd_hex):
        logger.error("보정 명령 전송 실패")
        return False
    logger.info(f"보정 명령 전송: {calib_cmd_hex}")

    # 보정은 보드 내부 처리에 시간이 걸리므로 충분히 대기(Hand_UI와 동일하게 1초)
    time.sleep(1.0)
    response = read_serial_data(ser, TIMEOUT_CMD, RESP_HEAD_GENERAL)
    if not response:
        logger.error("보정 응답을 받지 못했습니다(타임아웃)")
        return False

    # AA55 일반 응답: 헤더(2)+예약(1)+func(1)+addr(2)+len(2)+data(n)+lrc(1)
    if len(response) < 9 or response[:2] != RESP_HEAD_GENERAL:
        logger.error(f"보정 응답 헤더 오류: {response.hex()}")
        return False
    data_len = int.from_bytes(response[6:8], "little")
    data = response[8:8 + data_len]
    # 응답 데이터가 모두 0이면 성공(Hand_UI의 data == b"\x00" 판정과 동일)
    if data_len > 0 and all(b == 0 for b in data):
        logger.info("보정 성공")
        return True
    logger.error(f"보정 실패, 응답 데이터: {data.hex()}")
    return False

def parse_auto_push_data(data: bytes, expected_length: int = 0) -> Optional[Dict]:
    """자동 전송 데이터 프레임을 파싱합니다(AA56 헤더+예약+유효 프레임 길이+전체 오류 코드+유효 데이터+LRC)."""
    try:
        # 기본 검사: 최소 프레임 길이(Head2+예약1+유효 프레임 길이2+전체 오류 코드1+LRC1=7바이트)
        if len(data) < 7:
            logger.error(f"자동 전송 프레임이 너무 짧음: {len(data)}바이트 | 데이터: {data.hex()}")
            return None
        
        # 프레임 헤더 검사
        if data[:2] != AUTO_PUSH_FRAME_HEAD:
            logger.error(f"자동 전송 프레임 헤더 오류: {data[:2].hex()} 예상{AUTO_PUSH_FRAME_HEAD.hex()} | 데이터: {data.hex()}")
            return None
        
        # 기본 필드 추출
        parsed = {
            "head": data[:2].hex(),
            "reserved": data[2],
            "valid_frame_len": int.from_bytes(data[3:5], "little"),  # 유효 프레임 길이: 리틀 엔디언
            "error_code": data[5],
            "valid_data": b"",
            "valid_data_len": 0,
            "expected_data_len": expected_length,
            "length_match": False,
            "lrc_valid": False,
            "lrc_calc": 0,
            "lrc_recv": data[-1] if len(data) >= 7 else 0
        }
        
        # 유효 데이터 길이 계산(유효 프레임 길이 = 유효 데이터 길이 + 1)
        parsed["valid_data_len"] = parsed["valid_frame_len"] - 1
        
        # 유효 데이터 추출
        if parsed["valid_data_len"] > 0:
            data_end_pos = 6 + parsed["valid_data_len"]
            if data_end_pos <= len(data) - 1:  # LRC 위치 예약
                parsed["valid_data"] = data[6:data_end_pos]
            else:
                parsed["valid_data"] = data[6:-1]  # LRC 앞까지 잘라냄
                logger.warning(f"자동 전송 데이터가 불완전함: 예상{parsed['valid_data_len']}바이트, 실제{len(parsed['valid_data'])}바이트")
        
        # 데이터 길이가 예상 설정과 맞는지 확인
        if expected_length > 0:
            parsed["length_match"] = (parsed["valid_data_len"] == expected_length)
            if not parsed["length_match"]:
                logger.warning(f"데이터 길이가 설정과 맞지 않음: 실제{parsed['valid_data_len']}바이트, 예상{expected_length}바이트")
        
        # LRC 검사
        if len(data) >= 6 + parsed["valid_data_len"] + 1:
            parsed["lrc_calc"] = calc_lrc(data[:-1])
            parsed["lrc_valid"] = (parsed["lrc_calc"] == parsed["lrc_recv"])
            if not parsed["lrc_valid"]:
                logger.warning(f"자동 전송 LRC 검사 실패: 계산값0x{parsed['lrc_calc']:02X}, 실제0x{parsed['lrc_recv']:02X}")
        
        return parsed
    except Exception as e:
        logger.error(f"자동 전송 파싱 오류: {str(e)} | 데이터: {data.hex()}", exc_info=True)
        return None


def get_device_version(ser: serial.Serial) -> Optional[str]:
    """장치 버전 번호를 가져옵니다."""
    version_data = read_register(ser, VERSION_REG, VERSION_DATA_LEN)
    if version_data:
        try:
            ascii_version = version_data.decode("ascii", errors="ignore").strip()
            return f"ASCII: {ascii_version} | 16진수: {version_data.hex().upper()}"
        except:
            return f"16진수: {version_data.hex().upper()}"
    return None


def monitor_auto_push(ser: serial.Serial, duration: Optional[float] = None) -> None:
    """자동 전송 데이터를 모니터링합니다(모듈 설정은 사용하지 않음)."""
    start_time = time.time()
    logger.info(f"자동 전송 모니터링 시작")
    print("\n===== 자동 전송 데이터 수신 시작 =====")
    print(f"Ctrl+C로 중지")
    print("-" * 60)
    
    try:
        while True:
            # 모니터링 시간 확인
            if duration and time.time() - start_time > duration:
                logger.info(f"모니터링 타임아웃({duration}초)")
                break
            
            # 자동 전송 데이터 읽기
            push_data = read_serial_data(ser, TIMEOUT_AUTO_PUSH, AUTO_PUSH_FRAME_HEAD)
            if push_data:
                parsed = parse_auto_push_data(push_data)
                if parsed:
                    print(f"[{time.strftime('%H:%M:%S')}]")
                    print(f"  프레임 헤더: {parsed['head']} | 오류 코드: 0x{parsed['error_code']:02X}")
                    print(f"  데이터 길이: {parsed['valid_data_len']}바이트" )
                    print(f"  데이터 내용: {parsed['valid_data'].hex().upper()}")
                    print(f"  LRC검사: {'통과' if parsed['lrc_valid'] else '실패'}")
                    print("-" * 60)
    except KeyboardInterrupt:
        logger.info("사용자가 모니터링을 중단함")
        print("\n사용자가 수동으로 모니터링을 중지함")
    except Exception as e:
        logger.error(f"모니터링 오류: {str(e)}", exc_info=True)
        print(f"\n모니터링 오류: {str(e)}")


# -------------------------- Python main reusable API START --------------------------
def open_sensor_serial(port: str, baudrate: int = BAUDRATE, timeout: float = 1.0) -> serial.Serial:
    """Open the GEN3 high-speed communication board UART port."""
    ser = serial.Serial(
        port=port,
        baudrate=baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=timeout,
        write_timeout=0.5,
        inter_byte_timeout=0.001,
    )
    if not ser.is_open:
        ser.open()
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser


def frame_total_length(buffer: bytes, expected_head: bytes) -> Optional[int]:
    """Return full frame length when enough header/length bytes are available."""
    if expected_head == RESP_HEAD_GENERAL:
        if len(buffer) < 8:
            return None
        data_len = int.from_bytes(buffer[6:8], "little")
        return 8 + data_len + 1

    if expected_head == AUTO_PUSH_FRAME_HEAD:
        if len(buffer) < 5:
            return None
        valid_frame_len = int.from_bytes(buffer[3:5], "little")
        return 2 + 1 + 2 + valid_frame_len + 1

    return None


def read_protocol_frame(
    ser: serial.Serial,
    expected_head: bytes = AUTO_PUSH_FRAME_HEAD,
    timeout: float = TIMEOUT_AUTO_PUSH,
) -> Optional[bytes]:
    """Read one complete AA55 or AA56 protocol frame using the frame length field."""
    start_time = time.monotonic()
    recv_data = bytearray()

    while time.monotonic() - start_time < timeout:
        waiting = ser.in_waiting
        if waiting > 0:
            recv_data.extend(ser.read(waiting))
            head_pos = bytes(recv_data).find(expected_head)
            if head_pos >= 0:
                del recv_data[:head_pos]
            elif len(recv_data) > 1:
                del recv_data[:-1]

            if bytes(recv_data).startswith(expected_head):
                total_len = frame_total_length(bytes(recv_data), expected_head)
                if total_len is not None and len(recv_data) >= total_len:
                    return bytes(recv_data[:total_len])

            start_time = time.monotonic()
        else:
            time.sleep(0.001)

    return None


def make_timestamped_sample(
    seq: int,
    raw_frame: bytes,
    expected_length: int = 0,
    read_start_mono_ns: Optional[int] = None,
    read_end_mono_ns: Optional[int] = None,
) -> Dict:
    """Attach synchronization timestamps and parsed auto-push fields to a raw frame."""
    parsed = parse_auto_push_data(raw_frame, expected_length)
    host_time_ns = time.time_ns()
    if read_end_mono_ns is None:
        read_end_mono_ns = time.monotonic_ns()
    if read_start_mono_ns is None:
        read_start_mono_ns = read_end_mono_ns
    t_mono_ns = (int(read_start_mono_ns) + int(read_end_mono_ns)) // 2
    return {
        "seq": seq,
        "t_mono_ns": t_mono_ns,
        "t_wall_ns": host_time_ns,
        "read_start_mono_ns": int(read_start_mono_ns),
        "read_end_mono_ns": int(read_end_mono_ns),
        "host_time_ns": host_time_ns,
        "host_time_s": host_time_ns / 1_000_000_000,
        "monotonic_ns": t_mono_ns,
        "byte_count": len(raw_frame),
        "raw": raw_frame,
        "raw_hex": raw_frame.hex(" "),
        "parsed": parsed,
        "lrc_ok": None if parsed is None else parsed.get("lrc_valid"),
    }


def iter_auto_push_samples(
    ser: serial.Serial,
    expected_length: int = 0,
    timeout: float = TIMEOUT_AUTO_PUSH,
):
    """Yield timestamped auto-push samples from the sensor."""
    seq = 0
    while True:
        read_start_mono_ns = time.monotonic_ns()
        raw_frame = read_protocol_frame(ser, AUTO_PUSH_FRAME_HEAD, timeout)
        read_end_mono_ns = time.monotonic_ns()
        if raw_frame is None:
            continue
        yield make_timestamped_sample(
            seq,
            raw_frame,
            expected_length,
            read_start_mono_ns=read_start_mono_ns,
            read_end_mono_ns=read_end_mono_ns,
        )
        seq += 1


def start_auto_push_stream(ser: serial.Serial) -> bool:
    """Enable automatic data push from the sensor board."""
    return enable_auto_push(ser)


def stop_auto_push_stream(ser: serial.Serial) -> bool:
    """Disable automatic data push from the sensor board."""
    return disable_auto_push(ser)
# -------------------------- Python main reusable API END --------------------------

def main():
    logger.info("=== GEN3촉각 센서 프로그램 시작 ===")
    
    # 사용 가능한 시리얼 포트 검색
    available_ports = list(serial.tools.list_ports.comports())
    if not available_ports:
        logger.error("사용 가능한 시리얼 포트 없음")
        print("오류: 사용 가능한 시리얼 포트를 찾지 못했습니다")
        return
    
    # 시리얼 포트 선택
    print("\n사용 가능한 시리얼 포트: ")
    for i, port in enumerate(available_ports, 1):
        print(f"  {i}. {port.device} - {port.description}")
    
    try:
        choice = int(input(f"\n시리얼 포트 선택(1-{len(available_ports)}): "))
        selected_port = available_ports[choice - 1].device
    except (ValueError, IndexError):
        print("입력이 유효하지 않음, 첫 번째 시리얼 포트 선택")
        selected_port = available_ports[0].device
    
    # 시리얼 포트 초기화
    ser = None
    try:
        ser = serial.Serial(
            port=selected_port,
            baudrate=BAUDRATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1,
            write_timeout=0.5,
            inter_byte_timeout=0.001  # 고속 통신 중 바이트 손실 방지
        )
        
        if not ser.is_open:
            ser.open()
        print(f"\n시리얼 포트 연결: {ser.name}(baudrate{BAUDRATE})")
        
        # 버전 번호 명령 전송
        print("\n버전 번호 명령 전송...")
        version_cmd = "55AA000300000F00EF"
        if send_hex_cmd(ser, version_cmd):
            print("버전 번호 명령 전송 성공, 응답 대기...")
            response = read_serial_data(ser, TIMEOUT_CMD, RESP_HEAD_GENERAL)
            if response:
                print(f"버전 번호 응답 수신: {response.hex().upper()}")
                # 버전 번호 파싱 시도
                parsed = parse_response(response)
                if parsed and not parsed["is_error"]:
                    try:
                        ascii_version = parsed["data"].decode("ascii", errors="ignore").strip()
                        print(f"버전 번호 파싱: ASCII: {ascii_version}")
                    except:
                        print(f"버전 번호 파싱: 16진수: {parsed['data'].hex().upper()}")
            else:
                print("버전 번호 응답을 받지 못했습니다")
        else:
            print("버전 번호 명령 전송 실패")

        time.sleep(1)  

        # 자동 전송 활성화 명령 전송
        print("\n자동 전송 활성화 명령 전송...")
        auto_push_cmd = "55AA00101700010001D8"
        if send_hex_cmd(ser, auto_push_cmd):
            print("자동 전송 활성화 명령 전송 성공, 응답 대기...")
            response = read_serial_data(ser, TIMEOUT_CMD, RESP_HEAD_AUTO_PUSH)
            if response:
                print(f"자동 전송 응답 수신: {response.hex().upper()}")
                # 응답 파싱 시도
                parsed = parse_auto_response(response)
                if parsed and parsed["error_code"] == 0:
                    print("자동 전송이 성공적으로 켜졌습니다")
                    # 직접 자동 전송 수신
                    monitor_auto_push(ser)
                else:
                    print(f"자동 전송 켜기 실패, 오류 코드: 0x{parsed['error_code']:02X}" if parsed else "자동 전송 켜기 실패")
            else:
                print("자동 전송 명령 응답을 받지 못했습니다")
        else:
            print("자동 전송 활성화 명령 전송 실패")
    
        time.sleep(0.1) 
    
    except serial.SerialException as e:
        logger.error(f"시리얼 포트 오류: {str(e)}")
        print(f"\n시리얼 포트 오류: {str(e)}")
    except KeyboardInterrupt:
        print("\n사용자가 프로그램을 중단함")
    except Exception as e:
        logger.error(f"프로그램 오류: {str(e)}", exc_info=True)
        print(f"\n프로그램 오류: {str(e)}")
    finally:
        if ser and ser.is_open:
            print("\n자동 전송 비활성화 명령 전송...")
            disable_success = disable_auto_push(ser)
            if disable_success:
                print("자동 전송 비활성화 명령 전송 완료")
            else:
                print("자동 전송 비활성화 명령 전송 실패")
            ser.close()
            print("시리얼 포트가 닫혔습니다")
    
    logger.info("=== 프로그램 종료 ===")


if __name__ == "__main__":
    main()
