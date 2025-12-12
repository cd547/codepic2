import os
import shutil
import time
import cv2
from config import STORAGE_ROOT, IMAGE_QUALITY, SEPARATOR
from logger import init_logger

logger = init_logger()


def save_image(frame, valid_barcodes):
    print("开始存储图片...")
    temp_path = os.path.join(STORAGE_ROOT, f"temp_{int(time.time())}.jpg")
    os.makedirs(STORAGE_ROOT, exist_ok=True)
    cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])
    success = 0; fails = []
    seen = set()
    for barcode in valid_barcodes:
        data = barcode["data"]
        if data in seen:
            continue
        seen.add(data)
        try:
            order_part = data.split(SEPARATOR,1)[0]
            order_dir = os.path.join(STORAGE_ROOT, order_part)
            os.makedirs(order_dir, exist_ok=True)
            target = os.path.join(order_dir, f"{data}.jpg")
            if os.path.exists(target):
                fails.append(f"{data}:exists")
                continue
            shutil.copy2(temp_path, target)
            success += 1
            logger.info(f"存储 | {data} | 成功 | {target}")
        except Exception as e:
            fails.append(f"{data}:err")
            logger.error(f"存储异常 {data} {e}")
    os.remove(temp_path)
    if success == len(valid_barcodes):
        return "全部成功", success, 0
    elif success > 0:
        return f"部分成功（{';'.join(fails)}）", success, len(fails)
    else:
        return f"全部失败（{';'.join(fails)}）", 0, len(fails)
