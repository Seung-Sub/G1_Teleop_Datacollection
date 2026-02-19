import os
import numpy as np
import threading
import time
import atexit

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


from dynamixel_sdk import *                    # Uses Dynamixel SDK library

#********* DYNAMIXEL Model definition *********
MY_DXL = 'X_SERIES' 

# Control table address
ADDR_TORQUE_ENABLE          = 64
ADDR_GOAL_POSITION          = 116
LEN_GOAL_POSITION           = 4         # Data Byte Length
ADDR_PRESENT_POSITION       = 132
LEN_PRESENT_POSITION        = 4         # Data Byte Length
DXL_MINIMUM_POSITION_VALUE  = 0         # Refer to the Minimum Position Limit of product eManual
DXL_MAXIMUM_POSITION_VALUE  = 4095      # Refer to the Maximum Position Limit of product eManual
BAUDRATE                    = 1000000

OPERATING_MODE = 11
CURRENT_CONTROL_MODE = 0
POSITION_CONTROL_MODE = 3

ADDR_GOAL_CURRENT           = 102
LEN_GOAL_CURRENT            = 2         # Data Byte Length
ADDR_PRESENT_CURRENT        = 126
LEN_PRESENT_CURRENT         = 2         # Data Byte Length
DXL_CURRENT_LIMIT           = 100


PROTOCOL_VERSION            = 2.0

DXL1_ID                     = 1                 # Dynamixel#1 ID : 1
DXL2_ID                     = 2                 # Dynamixel#1 ID : 2

DEVICENAME                  = '/dev/ttyUSB0'

TORQUE_ENABLE               = 1                 # Value for enabling the torque
TORQUE_DISABLE              = 0                 # Value for disabling the torque
DXL_MOVING_STATUS_THRESHOLD = 20                # Dynamixel moving status threshold

TICKS_PER_REV    = 4096            # 0 ~ 4095 → 360°
TICKS_PER_RAD    = TICKS_PER_REV / (2 * np.pi)   # 라디안 당 틱 수
ZERO_OFFSET_TICK = TICKS_PER_REV // 2             # 180° 위치 = 2048


Num_Motors = 2

class MotorState:
    def __init__(self):
        self.q = None

class Dynamixel_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(Num_Motors)]


class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data

class Dynamixel_Controller:
    def __init__(self):
        print("Initialize Dynamixel")
        #self.q_target = np.array([2048, 2048])
        self.q_target = np.array([2048, 1934])
        self.port_lock = threading.Lock()

        self.running = True  # ← 플래그 추가


        self.motor_ids = [DXL1_ID, DXL2_ID]

        self.all_motor_q = None
        self.velocity_limit = 20.0
        self.control_dt = 1.0 / 100.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None


        self.portHandler = PortHandler(DEVICENAME)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)
        # Initialize GroupSyncWrite instance
        self.groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        # self.groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_GOAL_CURRENT, LEN_GOAL_CURRENT)
        # Initialize GroupSyncRead instace for Present Position
        self.groupSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
        # self.groupSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT)
        self.groupSyncRead.clearParam()

        self.lowstate_buffer = DataBuffer()



        self.initialize_dynamixel()
        self.operation_mode()
        time.sleep(0.01)

        self.enable_dynamixel_torque()
        self.add_parameter_storage()


        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while self.lowstate_buffer.GetData() is None:
            time.sleep(0.001)

        self.all_motor_q = self.get_current_motor_q()
        logger_mp.info(f"Current all body motor state q:\n{self.all_motor_q} \n")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize Dynamixel_Controller OK!\n")

        atexit.register(self.disconnect_dynamixel)



    def initialize_dynamixel(self):
        # Open port
        if not self.portHandler.openPort():
            raise RuntimeError("Failed to open the port")
        if not self.portHandler.setBaudRate(BAUDRATE):
            raise RuntimeError("Failed to change the baudrate")

    def operation_mode(self):
        for idx, dxl_id in enumerate(self.motor_ids):

            # Enable Dynamixel Torque
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, OPERATING_MODE, POSITION_CONTROL_MODE)
            if dxl_comm_result != COMM_SUCCESS:
                logger_mp.info("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                logger_mp.info("%s" % self.packetHandler.getRxPacketError(dxl_error))
            else:
                logger_mp.info("Dynamixel#%d has been position control mode" % dxl_id)

    def enable_dynamixel_torque(self):
        for idx, dxl_id in enumerate(self.motor_ids):

            # Enable Dynamixel Torque
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
            if dxl_comm_result != COMM_SUCCESS:
                logger_mp.info("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                logger_mp.info("%s" % self.packetHandler.getRxPacketError(dxl_error))
            else:
                logger_mp.info("Dynamixel#%d has been successfully connected" % dxl_id)

    def add_parameter_storage(self):
        for dxl_id in self.motor_ids:
            if not self.groupSyncRead.addParam(dxl_id):
                raise RuntimeError(f"[ID:{dxl_id:03d}] groupSyncRead addparam failed")


    def _subscribe_motor_state(self):
        loop_count = 0
        t_start = time.perf_counter()
        while self.running:
            with self.port_lock:
                res = self.groupSyncRead.txRxPacket()
            if res != COMM_SUCCESS:
                logger_mp.info(self.packetHandler.getTxRxResult(res))

            for dxl_id in self.motor_ids:
                if not self.groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                # if not self.groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT):
                    logger_mp.info(f"[ID:{dxl_id:03d}] groupSyncRead getdata failed")
                    quit()

            lowstate = Dynamixel_LowState()
            for idx, dxl_id in enumerate(self.motor_ids):
                lowstate.motor_state[idx].q = self.groupSyncRead.getData(
                    dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                    # dxl_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT)
            self.lowstate_buffer.SetData(lowstate)

            # 주기 측정
            loop_count += 1
            now = time.perf_counter()
            if now - t_start >= 1.0:
                # logger_mp.info(f"[Subscribe] {loop_count/(now-t_start):.1f} Hz")
                loop_count = 0
                t_start = now

            time.sleep(1/100)



    def clip_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_motor_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_q_target
    

    def _ctrl_motor_state(self):
        loop_count = 0
        t_start = time.perf_counter()
        while self.running:
            start_time = time.time()

            with self.ctrl_lock:
                q_target = self.q_target
                q_target = np.array([2048, 1934])
                #q_target = np.array([2048, 2048])
            # cliped_q_target = self.clip_q_target(q_target, velocity_limit = self.velocity_limit)


            for idx, dxl_id in enumerate(self.motor_ids):
                # 1) 스칼라 → int
                goal = int(round(q_target[idx]))

                # 2) SDK 매크로로 4바이트 분리
                param_goal_position = [
                    DXL_LOBYTE(DXL_LOWORD(goal)),
                    DXL_HIBYTE(DXL_LOWORD(goal)),
                    DXL_LOBYTE(DXL_HIWORD(goal)),
                    DXL_HIBYTE(DXL_HIWORD(goal)),
                ]
                
                # 3) 파라미터 등록
                dxl_addparam_result = self.groupSyncWrite.addParam(dxl_id, param_goal_position)
                # dxl_addparam_result = self.groupSyncWrite.addParam(dxl_id, param_goal_current)
                if not dxl_addparam_result:
                    logger_mp.info(f"[ID:{dxl_id:03d}] groupSyncWrite addparam failed")
                    raise RuntimeError("SyncWrite addParam failed")

            with self.port_lock:
                dxl_comm_result = self.groupSyncWrite.txPacket()
            if dxl_comm_result != COMM_SUCCESS:
                logger_mp.info("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

            # Clear syncwrite parameter storage
            self.groupSyncWrite.clearParam()

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            loop_count += 1
            now = time.perf_counter()
            if now - t_start >= 1.0:
                # logger_mp.info(f"[Control ] {loop_count/(now-t_start):.1f} Hz")
                loop_count = 0
                t_start = now


            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)


    def ctrl_dynamixel(self, q_target):
        '''Set control target values qof the motors.'''
        with self.ctrl_lock:
            self.q_target = q_target

    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        lowstate = self.lowstate_buffer.GetData()
        return np.array([ms.q for ms in lowstate.motor_state])


#----------------------------------------------------------------------#

    def ctrl_dynamixel_go_home(self):
        logger_mp.info("[Dynamixel Controller] ctrl_dynamixel_go_home start...")
        with self.ctrl_lock:
            self.q_target = np.array([2048, 2048])
            # self.q_target = np.array([0, 0])

        tolerance = 1  # 허용 오차
        max_iter = 300    # 무한 루프 방지를 위한 안전 장치
        iter_count = 0

        while True:
            current_q = self.get_current_motor_q()

            # 실제 제어 명령 전송
            self.ctrl_dynamixel(self.q_target)

            if np.all(np.abs(current_q - self.q_target) < tolerance):
                logger_mp.info("[Dynamixel Controller] both dynamixel have reached the home position.")
                break

            iter_count += 1
            if iter_count > max_iter:
                logger_mp.info("[Dynamixel Controller] Home movement timeout.")
                break

            time.sleep(0.05)



    def rad_to_dxl_tick(self, q_rad):
        """
        q_rad: 입력 각도 [rad], 0 → 그림의 180° 방향
        반환값: 0~4095 범위로 클립된 goal position 틱값 (int)
        """
        tick = ZERO_OFFSET_TICK + q_rad * TICKS_PER_RAD
        tick = int(round(tick))
        # 서보 허용 범위로 클립
        return max(0, min(4095, tick))
    
    def dxl_tick_to_rad(self, tick):
        """
        tick: 입력 목표 위치 틱값 [0~4095]
        반환값: 라디안 [rad], 0 → 그림의 180° 방향
        """
        # 정수로 반올림 후 서보 허용 범위로 클립
        tick = int(round(tick))
        tick = max(0, min(TICKS_PER_REV - 1, tick))
        # 라디안으로 변환
        q_rad = (tick - ZERO_OFFSET_TICK) / TICKS_PER_RAD
        return q_rad

#----------------------------------------------------------------------#

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for dynamixel velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set dynamixel velocity to the maximum value immediately, instead of gradually increasing.'''
        self.velocity_limit = 30.0

#----------------------------------------------------------------------#

    def disconnect_dynamixel(self):
        self.running = False
        self.subscribe_thread.join(timeout=1.0)
        self.publish_thread.join(timeout=1.0)

        for dxl_id in self.motor_ids:
            try:
                self.packetHandler.write1ByteTxRx(
                    self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
                )
            except Exception as e:
                logger_mp.info(f"[Warning] ID:{dxl_id:03d} torque disable failed: {e}")

        # 4) 포트 닫기
        try:
            self.portHandler.closePort()
        except Exception as e:
            logger_mp.info(f"[Warning] Port close failed: {e}")

            
    def __del__(self):
        """객체 소멸 시에도 안전하게 disconnect 호출"""
        try:
            self.disconnect_dynamixel()
        except Exception:
            pass
















