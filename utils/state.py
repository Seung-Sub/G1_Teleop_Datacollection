from enum import Enum, auto

class State(Enum):
    WAIT_CONNECT = auto()
    WAIT_START   = auto()
    RUN          = auto()
    PAUSE        = auto()
    EXIT         = auto()

class EventsSnapshot:
    def __init__(self, *, shutdown: bool, emergency: bool,
                 g1_ready: bool, hand_ready: bool, start: bool, select_pressed: bool):
        self.shutdown = shutdown
        self.emergency = emergency
        self.g1_ready = g1_ready
        self.hand_ready = hand_ready
        self.start = start
        self.select_pressed = select_pressed

def next_state(state: State, e: EventsSnapshot) -> State:
    # 글로벌 종료 우선
    if e.shutdown or e.emergency:
        return State.EXIT

    if state == State.WAIT_CONNECT:
        return State.WAIT_START if e.g1_ready and e.hand_ready else State.WAIT_CONNECT

    if state == State.WAIT_START:
        return State.RUN if e.start else State.WAIT_START

    if state == State.RUN:
        # start 해제되거나 select가 눌리면 일시정지
        if not e.start or e.select_pressed:
            return State.PAUSE
        return State.RUN

    if state == State.PAUSE:
        # 일시정지에서 start가 다시 눌리지 않은 상태를 유지 → START 대기로
        return State.WAIT_START if not e.start else State.PAUSE

    return State.EXIT
