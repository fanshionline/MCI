import ctypes
import numpy as np
import cv2

thermal = ctypes.cdll.LoadLibrary(".\\GuideInfra.dll")
thermal.guide_init(384, 288, thermal.Y16_PARAM)

while True:
    data = np.ndarray((384, 288), int, thermal.read_data())
    print(data[0])
    frame = np.ndarray((384, 288, 3), int, thermal.read_frame())
    # cv2.imshow(frame)
    # cv2.waitKey(1)