"""
main.py - 精简主流程，调用模块化实现
此文件现在只负责进程控制与 UI，功能细分到 camera.py, decoder.py, display.py, storage.py, config.py, logger.py
"""
import time
import cv2
from logger import init_logger
from camera import check_camera, is_frame_static
from decoder import decode_barcode, update_barcode_cache, DecoderWorker
import queue
from display import draw_text_pil, draw_persistent_items
from storage import save_image
from config import *
import os
import glob
from pathlib import Path
logger = init_logger()

def validate_barcode(barcode_data):
    """校验条形码有效性（含订单号+序号）"""
    # 格式：前缀+数字+分隔符+数字
    print(f"barcode_data{barcode_data}")
    if SEPARATOR not in barcode_data:
        return False, "缺少分隔符"
    
    order_part, serial_part = barcode_data.split(SEPARATOR, 1)

    order_num = order_part.replace(ORDER_PREFIX, "")

    
    return True, "有效"


def _collect_test_images(tst_dir):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(tst_dir, e)))
    return sorted(files)



# -------------------------- 主逻辑 --------------------------
def main():
    cap = None
    test_images = None
    img_index = 0
    if TEST_MODE:
        tst_dir = TST_DIR
        test_images = _collect_test_images(tst_dir)
        if not test_images:
            logger.error(f"测试模式已启用，但目录未发现图片：{tst_dir}")
            return
        logger.info(f"测试模式：从 {tst_dir} 加载 {len(test_images)} 张图片")
    else:
        cap = check_camera(0)
        if cap is None:
            print("错误：摄像头不可用")
            return

    # 启动后台解码 Worker（主线程不再直接执行耗时解码）
    results_queue = queue.Queue(maxsize=4)
    worker = DecoderWorker(results_queue=results_queue, min_interval=0.5)
    worker.start()

    print("按 's' 开始/暂停，按 'q' 退出")
    recognition_enabled = True
    last_frame = None
    last_recognition_time = 0
    last_displayed = {"time":0, "items":[]}

    while True:
        # ret, frame = cap.read()
        # frame = cv2.imread('5.jpg')
        # if not ret:
        #     logger.error("识别 | 无 | 失败 | 图像采集失败")
        #     break
        if TEST_MODE:
            # 顺序读取测试图片；每张停留 0.5s，可按需求修改
            img_path = test_images[img_index]
            frame = cv2.imread(img_path)
            if frame is None:
                logger.warning(f"无法读取测试图片：{img_path}")
                img_index = (img_index + 1) % len(test_images)
                continue
            img_index = (img_index + 1) % len(test_images)
            ret = True
            # 确保每张测试图至少显示 DISPLAY_HOLD_SECONDS（与识别结果保持时间同步）
            time.sleep(max(0.5, DISPLAY_HOLD_SECONDS))
        else:
            ret, frame = cap.read()
            if not ret:
                logger.error("识别 | 无 | 失败 | 图像采集失败")
                break
        display = frame.copy()
        current_time = time.time()

        # 在测试模式下跳过静止检测（测试图片通常会变化），否则保留静止检测以减少误触发
        istestorisframestatic=TEST_MODE or is_frame_static(frame, last_frame)
        isRecInterva=(current_time - last_recognition_time) >= RECOGNITION_INTERVAL
        # print(f'current_time:{current_time}-last_recognition_time:{last_recognition_time}={current_time-last_recognition_time},RECOGNITION_INTERVAL{RECOGNITION_INTERVAL}')
        # print("istestorisframestatic",istestorisframestatic,"isRecInterva",isRecInterva)
        print(f'=>{recognition_enabled and istestorisframestatic and isRecInterva}')
        if recognition_enabled and istestorisframestatic and isRecInterva:
            last_recognition_time = current_time
            # 将帧提交给后台解码线程进行异步解码，避免主线程阻塞
            try:
                logger.debug(f"提交帧给后台解码线程")
                worker.put_frame(frame)
            except Exception:
                pass

        if time.time() - last_displayed["time"] <= DISPLAY_HOLD_SECONDS:
            display = draw_persistent_items(display, last_displayed["items"])

        status = f"RECG:{'ON' if recognition_enabled else 'OFF'}  DIR:{STORAGE_ROOT}"
        display = draw_text_pil(display, status, pos=(12,8), font_size=18, bg_color=(0,0,0))

        cv2.imshow("条形码识别与存储系统", display)
        # 处理后台解码结果（非阻塞）
        try:
            barcodes, decoded_frame = results_queue.get_nowait()
            logger.info(f"后台识别到 {len(barcodes)} 条码")
            if barcodes:
                print(f"barcodes：{barcodes}")
                confirmed = update_barcode_cache(barcodes)
                print(f"确认条码数：{len(confirmed)}")
                if confirmed:
                    valid = []
                    items = []
                    for b in confirmed:
                        ok, reason = validate_barcode(b["data"]) if 'validate_barcode' in globals() else (True, '有效')
                        items.append({"data":b["data"], "pos":b["pos"], "valid":ok})
                        if ok:
                            valid.append(b)
                    if items:
                        last_displayed["time"] = time.time()
                        last_displayed["items"] = items
                    if valid:
                        print(f"调用存储，有效条码数：{len(valid)}")
                        save_image(decoded_frame, valid)
        except queue.Empty:
            pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            logger.info("系统 | 无 | 成功 | 用户退出系统")
            break
        if key == ord("s"):
            recognition_enabled = not recognition_enabled
            logger.info(f"系统 | 无 | 成功 | 识别功能{'开启' if recognition_enabled else '关闭'}")
            time.sleep(0.3)

        last_frame = frame

    # cap.release()
    if cap is not None:
        cap.release()
    # 停止后台 worker
    try:
        worker.stop()
        worker.join(timeout=1.0)
    except Exception:
        pass
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = init_logger()
        logger.error(f"系统 | 无 | 失败 | 系统异常崩溃：{str(e)}")
        raise