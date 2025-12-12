import cv2
import time
import numpy as np
from logger import init_logger
from config import STATIC_THRESHOLD

logger = init_logger()


def check_camera(device=0, width=1280, height=720):
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        time.sleep(1)
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            logger.error("识别 | 无 | 失败 | 摄像头调用失败")
            return None
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    logger.info("识别 | 无 | 成功 | 摄像头调用成功")
    return cap


def is_frame_static(current_frame, last_frame):
    """判定图像是否静止（像素变化率低于阈值）。如果两帧尺寸不同，会先对齐尺寸。"""
    if last_frame is None:
        return False
    try:
        # 如果尺寸不同，缩放 last_frame 到 current_frame 大小以保证可比较
        if current_frame.shape[:2] != last_frame.shape[:2]:
            last_resized = cv2.resize(last_frame, (current_frame.shape[1], current_frame.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
        else:
            last_resized = last_frame

        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_last = cv2.cvtColor(last_resized, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_current, gray_last)
        non_zero = np.count_nonzero(diff)
        total = diff.shape[0] * diff.shape[1]
        change_rate = non_zero / total
        # print("change_rate",change_rate)
        return change_rate <= STATIC_THRESHOLD
    except Exception as e:
        logger.debug(f"is_frame_static 异常：{e}")
        # 出现异常时保守返回 False，避免阻塞识别流程
        return False
