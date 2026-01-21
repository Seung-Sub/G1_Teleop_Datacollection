import pyzed.sl as sl

zed = sl.Camera()
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.AUTO
init.camera_fps        = 30
init.depth_mode        = sl.DEPTH_MODE.NONE

if zed.open(init) != sl.ERROR_CODE.SUCCESS:
    raise RuntimeError("ZED 열기 실패")

runtime = sl.RuntimeParameters()
left = sl.Mat()
if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
    zed.retrieve_image(left, sl.VIEW.LEFT)
    print("프레임 캡처 성공, shape =", left.get_data().shape)

zed.close()
