# import sys
# import os
# import termios
# import tty
# import time

# import logging_mp
# logger_mp = logging_mp.get_logger(__name__)


# def _getch() -> str:
#     try:
#         fd = os.open("/dev/tty", os.O_RDONLY)
#         close_fd = True
#     except OSError:
#         fd = sys.stdin.fileno()
#         close_fd = False

#     old_attr = termios.tcgetattr(fd)
#     try:
#         tty.setraw(fd)
#         ch = os.read(fd, 1)
#         return ch.decode("utf-8", "ignore")
#     finally:
#         termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
#         if close_fd:
#             os.close(fd)


# def keyboard_listener(shared_event):


#     logger_mp.info("키 입력: [q] → 100Hz 워커 종료, [s] → 250Hz 워커 종료")

#     try:
#         while not shared_event["shutdown"].is_set():
#             key = _getch().lower()
#             if key == "q": # quit
#                 logger_mp.info("quit!")
#                 shared_event["shutdown"].set()
#                 break
#             elif key == "h": # go home
#                 logger_mp.info("go home!")
#                 shared_event["go_home"].set()
#             time.sleep(0.01)
#         logger_mp.warning("Close Process!")
#     except KeyboardInterrupt: # ctrl + c
#         shared_event["shutdown"].set()


import sys, os, termios, tty, time, atexit, signal
import logging_mp
logger_mp = logging_mp.get_logger(__name__)

# 전역 상태(한 번만 /dev/tty 또는 stdin fd를 정하고, 종료 시 복구)
_TTY_FD = None
_OLD_ATTR = None
_CLOSE_FD = False

def _open_tty_once():
    """컨트롤링 TTY fd와 원래 속성을 한번만 잡아둔다."""
    global _TTY_FD, _OLD_ATTR, _CLOSE_FD
    if _TTY_FD is not None:
        return
    try:
        _TTY_FD = os.open("/dev/tty", os.O_RDONLY)
        _CLOSE_FD = True
    except OSError:
        _TTY_FD = sys.stdin.fileno()
        _CLOSE_FD = False
    _OLD_ATTR = termios.tcgetattr(_TTY_FD)

def _restore_tty():
    """프로세스 종료 시에도 TTY 되돌리기"""
    global _TTY_FD, _OLD_ATTR, _CLOSE_FD
    try:
        if _TTY_FD is not None and _OLD_ATTR is not None:
            termios.tcsetattr(_TTY_FD, termios.TCSADRAIN, _OLD_ATTR)
    except Exception:
        pass
    finally:
        if _CLOSE_FD and _TTY_FD not in (None, getattr(sys.stdin, "fileno", lambda: None)()):
            try:
                os.close(_TTY_FD)
            except Exception:
                pass
        _TTY_FD = None
        _OLD_ATTR = None
        _CLOSE_FD = False

# 어떤 종료 경로든 복구 시도(※ kill -9는 복구 불가)
atexit.register(_restore_tty)
for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
    try:
        signal.signal(_sig, lambda *_: (_restore_tty(), sys.exit(0)))
    except Exception:
        pass  # 스레드/서브프로세스 환경 등에서 실패 가능

def _read_key_blocking() -> str:
    """raw 모드에서 1바이트 읽고 문자열로 반환(멀티바이트는 첫 바이트만)"""
    ch = os.read(_TTY_FD, 1)
    return ch.decode("utf-8", "ignore")

def keyboard_listener(shared_event):
    logger_mp.info("키 입력: [q] → 종료, [h] → go home")
    _open_tty_once()

    # 루프 전체를 raw로 감싸고, 마지막에 반드시 복구
    try:
        tty.setcbreak(_TTY_FD)  # 또는 tty.setcbreak(_TTY_FD)

        # 에코만 끄고, 출력 후처리는 유지/보강
        attrs = termios.tcgetattr(_TTY_FD)
        attrs[3] &= ~termios.ECHO     # lflag: 에코 끄기
        attrs[1] |= termios.OPOST     # oflag: 출력 후처리 켜기
        try:
            attrs[1] |= termios.ONLCR # '\n' → '\r\n' (플랫폼에 없을 수 있어 try/except)
        except AttributeError:
            pass
        # 읽기 블록 동작 보장
        attrs[6][termios.VMIN]  = 1
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(_TTY_FD, termios.TCSADRAIN, attrs)
        while not shared_event["shutdown"].is_set():
            key = _read_key_blocking()
            if not key:
                continue
            k = key.lower()
            if k == "q":
                logger_mp.info("quit!")
                shared_event["shutdown"].set()
                break
            elif k == "h":
                logger_mp.info("go home!")
                shared_event["go_home"].set()
            time.sleep(0.01)
        logger_mp.warning("Close Process!")
    except KeyboardInterrupt:
        shared_event["shutdown"].set()
    finally:
        # 여기서도 복구(정상/예외 모두)
        _restore_tty()
