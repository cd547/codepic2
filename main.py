import cv2
import pyzbar.pyzbar as pyzbar
from PIL import Image
import numpy as np
import os
import time
import logging
from dotenv import load_dotenv
import shutil
from datetime import datetime
import threading

# 加载配置文件
load_dotenv()

# -------------------------- 全局配置加载 --------------------------
# 识别参数
RECOGNITION_INTERVAL = float(os.getenv("RECOGNITION_INTERVAL", 0.4))
STATIC_THRESHOLD = float(os.getenv("STATIC_THRESHOLD", 10.0)) / 100  # 转为小数
SENSITIVITY = os.getenv("RECOGNITION_SENSITIVITY", "high")

# 数据格式
ORDER_PREFIX = os.getenv("ORDER_PREFIX", "")
ORDER_NUM_LENGTH = int(os.getenv("ORDER_NUM_LENGTH", 7))
SERIAL_NUM_LENGTH = int(os.getenv("SERIAL_NUM_LENGTH", 3))
SEPARATOR = os.getenv("SEPARATOR", "-")

# 存储配置
STORAGE_ROOT = os.getenv("STORAGE_ROOT", "D:\\Projects\\coderpic2\\pic\\")
IMAGE_QUALITY = int(os.getenv("IMAGE_QUALITY", 85))

# 新增：多帧验证参数（全局变量）
FRAME_VALIDATE_COUNT = 2  # 连续2帧识别到相同条形码才确认有效
barcode_cache = []  # 缓存格式：[{"data": "xxx", "count": 0, "timestamp": 0, "info": {}}]

# 初始化状态
is_running = False  # 识别状态标记
last_frame = None   # 上一帧图像（用于静止判定）
log_dir = os.path.join(STORAGE_ROOT, "Log")  # 日志目录

# -------------------------- 日志配置 --------------------------
def init_logger():
    """初始化日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
    
    logger = logging.getLogger("BarcodeSystem")
    logger.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    
    # 日志格式：时间戳|操作类型|条形码数据|处理结果|详细描述
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

logger = init_logger()

# -------------------------- 工具函数 --------------------------
def check_camera():
    """检查摄像头是否可用"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # 重试1次（共2次）
        time.sleep(1)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error(f"识别 | 无 | 失败 | 摄像头调用失败：未检测到可用摄像头或被占用")
            return None
    
    # 关键优化：开启自动对焦（部分摄像头支持）
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 1=开启自动对焦
    # 优化曝光（避免过暗/过亮）
    cap.set(cv2.CAP_PROP_EXPOSURE, -4)  # -4=自动曝光（可根据实际调整，范围-10~10）
    # 优化白平衡
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # 1=自动白平衡
    # 提高采集帧率（减少模糊）
    cap.set(cv2.CAP_PROP_FPS, 30)
    logger.info(f"识别 | 无 | 成功 | 摄像头调用成功")
    return cap

def is_frame_static(current_frame, last_frame):
    """判定图像是否静止（像素变化率低于阈值）"""
    if last_frame is None:
        return False
    # 转为灰度图计算差异
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_last = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    # 计算差异像素数
    diff = cv2.absdiff(gray_current, gray_last)
    non_zero_count = np.count_nonzero(diff)
    total_pixels = diff.shape[0] * diff.shape[1]
    change_rate = non_zero_count / total_pixels
    return change_rate <= STATIC_THRESHOLD


def decode_barcode(frame):
    """解码图像中的CODE128条形码（专属优化：识别率+效率双提升）"""
    results = []
    processed_frames = []
    
    # -------------------------- CODE128专属增强函数 --------------------------
    # 1. 锐化（强化水平线条对比，CODE128核心需求）
    def sharpen_image(img):
        # 水平方向锐化核（更适配CODE128的水平黑白条）
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)
    
    # 2. 旋转矫正（重点处理水平倾斜，CODE128多为水平方向）
    def correct_rotation(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        # 优化边缘检测：侧重水平边缘（CODE128以水平线条为主）
        edges = cv2.Canny(gray, 40, 130, apertureSize=3)
        # 霍夫直线检测：过滤短线条，聚焦长水平线条
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=90, minLineLength=80, maxLineGap=15)
        if lines is None:
            return img
        
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 只保留接近水平的线条（角度±30°内）
            if x2 - x1 == 0:
                continue
            angle = np.arctan2((y2 - y1), (x2 - x1)) * 180 / np.pi
            if -30 < angle < 30 and abs(angle) > 0.5:
                angles.append(angle)
        
        if not angles:
            return img
        
        # 去除异常值，确保角度精准
        angles = np.array(angles)
        mean_angle = np.mean(angles)
        valid_angles = angles[np.abs(angles - mean_angle) < 1.5 * np.std(angles)]
        final_angle = np.mean(valid_angles) if len(valid_angles) > 0 else mean_angle
        
        # 角度小于1°不矫正（避免失真）
        if abs(final_angle) < 1.0:
            return img
        
        # 旋转图像（保持水平方向完整，避免CODE128被裁剪）
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, final_angle, 1.0)
        # 计算新尺寸，确保旋转后图像不裁剪
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
# 修复：正确设置边界填充值
        corrected = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_CUBIC, 
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))        
        logger.debug(f"识别 | 无 | 信息 | CODE128倾斜矫正：{final_angle:.2f}度")
        return corrected
    
    # 3. 局部直方图均衡化（适配CODE128低光照/打印模糊场景）
    def clahe_equalize(img):
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))  # 提高clipLimit，增强对比
        return clahe.apply(img)
    
    # 4. CODE128专属形态学操作（增强水平线条连续性）
    def morph_enhance(img):
        # 水平方向膨胀核（填充水平线条空隙）
        kernel_horizontal = np.ones((1, 3), np.uint8)
        # 垂直方向腐蚀核（去除垂直噪点）
        kernel_vertical = np.ones((3, 1), np.uint8)
        
        # 先膨胀再腐蚀，增强水平线条
        morph = cv2.dilate(img, kernel_horizontal, iterations=1)
        morph = cv2.erode(morph, kernel_vertical, iterations=1)
        # 闭运算填充小空隙
        kernel_close = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        return morph
    
    # -------------------------- 预处理流程（CODE128专属优化）--------------------------
    # 第一步：旋转矫正（优先解决水平倾斜）
    base_frame = correct_rotation(frame.copy())
    # 第二步：转为灰度图（CODE128无需彩色信息，减少计算）
    gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY) if len(base_frame.shape) == 3 else base_frame
    # 第三步：增强对比度+局部均衡化（解决打印模糊/低光照）
    enhanced_gray = cv2.convertScaleAbs(gray, alpha=1.6, beta=25)  # 提高对比度参数
    enhanced_gray = clahe_equalize(enhanced_gray)
    
    # 第四步：生成CODE128专属预处理组合（仅保留最有效的4种）
    # 1. 增强后灰度图+锐化（基础场景）
    processed_frames.append(sharpen_image(enhanced_gray))
    # 2. OTSU自动阈值（适配大部分光照）
    _, thresh_otsu = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_frames.append(thresh_otsu)
    # 3. 形态学增强+锐化（模糊/有噪点场景）
    morph_img = morph_enhance(thresh_otsu)
    processed_frames.append(sharpen_image(morph_img))
    # 4. 自适应阈值（局部光照不均场景）
    adaptive = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 27, 6)
    processed_frames.append(adaptive)
    
    # -------------------------- 锁定CODE128类型解码 --------------------------
    # 只保留CODE128符号类型（无其他类型冗余）
    symbol_type = None
    try:
        symbol_type = pyzbar.ZBarSymbol.CODE128  # 直接指定CODE128类型
    except AttributeError:
        logger.error(f"识别 | 无 | 失败 | 不支持CODE128类型识别，请更新pyzbar库")
        return results
    
   # 修改核心预处理解码部分（按效果优先级遍历，收集所有识别结果）
    all_barcodes = []  # 收集所有识别到的条形码
    for i, img in enumerate(processed_frames):
        try:
            barcodes = pyzbar.decode(img, symbols=[symbol_type])  # 只解码CODE128
            if barcodes:
                all_barcodes.extend(barcodes)  # 收集所有识别结果
                logger.debug(f"使用CODE128专属预处理方式 {i} 识别到 {len(barcodes)} 个条形码")
        except Exception as e:
            logger.debug(f"预处理方式 {i} 解码失败：{str(e)}")
            continue
    
    # 小尺寸CODE128适配（多比例缩放+二次增强）
    if not all_barcodes:  # 只有在完全没有识别到时才尝试缩放
        logger.debug(f"常规预处理未识别到，尝试CODE128专属缩放")
        scales = [1.3, 1.6, 1.9]  # 适配CODE128常见小尺寸（避免过度缩放）
        for scale in scales:
            try:
                h, w = base_frame.shape[:2]
                new_w, new_h = int(w * scale), int(h * scale)
                scaled = cv2.resize(base_frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                # 缩放后二次增强（针对小尺寸模糊）
                scaled_gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
                scaled_gray = clahe_equalize(scaled_gray)
                scaled_thresh = cv2.threshold(scaled_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                scaled_final = sharpen_image(morph_enhance(scaled_thresh))
                
                barcodes = pyzbar.decode(scaled_final, symbols=[symbol_type])
                if barcodes:
                    all_barcodes.extend(barcodes)  # 收集所有识别结果
                    logger.debug(f"CODE128缩放 {scale} 倍后识别到 {len(barcodes)} 个条形码")
            except Exception as e:
                logger.warning(f"缩放 {scale} 倍解码异常：{str(e)}")
                continue
    
    # -------------------------- 结果解析（去重+位置修正）--------------------------
    for barcode in all_barcodes:
        try:
            # 只保留CODE128类型结果（双重保险）
            if barcode.type != "CODE128":
                continue
            
            barcode_data = barcode.data.decode("utf-8").strip()
            if not barcode_data:
                continue
            
            # 位置信息
            (x, y, w, h) = barcode.rect
            
            # 去重（CODE128数据+位置容差25像素，适配轻微位移）
            if not any(
                r["data"] == barcode_data and
                abs(r["pos"][0] - x) < 25 and
                abs(r["pos"][1] - y) < 25
                for r in results
            ):
                results.append({
                    "data": barcode_data,
                    "pos": (x, y, w, h),
                    "type": "CODE128"
                })
        except Exception as e:
            logger.warning(f"CODE128解码异常（数据：{barcode.data}）：{str(e)}")
    
    # 移除多帧验证机制，直接返回识别结果（针对图片处理）
    logger.debug(f"识别 | 无 | 信息 | CODE128识别数量：{len(results)}")
    return results

def validate_barcode(barcode_data):
    """校验条形码有效性（含订单号+序号）"""
    # 格式：前缀+数字+分隔符+数字
    print(f"barcode_data{barcode_data}")
    if SEPARATOR not in barcode_data:
        return False, "缺少分隔符"
    
    order_part, serial_part = barcode_data.split(SEPARATOR, 1)
    # 校验订单号
    # if not order_part.startswith(ORDER_PREFIX):
    #     return False, f"订单号前缀错误（需为{ORDER_PREFIX}）"
    order_num = order_part.replace(ORDER_PREFIX, "")
    # if not order_num.isdigit() or len(order_num) != ORDER_NUM_LENGTH:
    #     return False, f"订单号格式错误（需{ORDER_NUM_LENGTH}位数字）"
    # 校验序号
    # if not serial_part.isdigit() or len(serial_part) != SERIAL_NUM_LENGTH:
    #     return False, f"序号格式错误（需{SERIAL_NUM_LENGTH}位数字）"
    # if not (1 <= int(serial_part) <= 10**SERIAL_NUM_LENGTH - 1):
    #     return False, f"序号超出范围（1-{10**SERIAL_NUM_LENGTH - 1}）"
    
    return True, "有效"

def save_image(frame, valid_barcodes):
    """按规则存储图像（去重+多订单关联）"""
    # 生成图像临时文件（避免重复存储）
    temp_path = os.path.join(STORAGE_ROOT, f"temp_{int(time.time())}.jpg")
    cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])
    
    success_count = 0
    fail_reasons = []
    order_serials = []  # 已处理的订单号+序号组合（去重）
    
    for barcode in valid_barcodes:
        barcode_data = barcode["data"]
        order_part, serial_part = barcode_data.split(SEPARATOR, 1)
        order_num = order_part  # 完整订单号（含前缀）
        file_name = f"{barcode_data}.jpg"
        order_dir = os.path.join(STORAGE_ROOT, order_num)
        
        # 跳过重复的订单号+序号组合
        if barcode_data in order_serials:
            continue
        order_serials.append(barcode_data)
        
        try:
            # 创建订单子目录
            os.makedirs(order_dir, exist_ok=True)
            target_path = os.path.join(order_dir, file_name)
            
            # 检查文件是否已存在
            if os.path.exists(target_path):
                fail_reasons.append(f"{barcode_data}：文件已存在")
                continue
            
            # 复制临时文件到目标路径（避免重复编码）
            shutil.copy2(temp_path, target_path)
            success_count += 1
            logger.info(f"存储 | {barcode_data} | 成功 | 图像已保存至{target_path}")
        except PermissionError:
            fail_reasons.append(f"{barcode_data}：无写入权限")
            logger.error(f"存储 | {barcode_data} | 失败 | 无写入权限：{order_dir}")
        except Exception as e:
            fail_reasons.append(f"{barcode_data}：未知错误")
            logger.error(f"存储 | {barcode_data} | 失败 | 存储异常：{str(e)}")
    
    # 删除临时文件
    os.remove(temp_path)
    
    # 返回存储结果
    if success_count == len(valid_barcodes):
        return "全部成功", success_count, 0
    elif success_count > 0:
        return f"部分成功（失败原因：{'; '.join(fail_reasons)}）", success_count, len(fail_reasons)
    else:
        return f"全部失败（原因：{'; '.join(fail_reasons)}）", 0, len(fail_reasons)

# -------------------------- 主逻辑 --------------------------
def main():
    global is_running, last_frame
    is_running = True
    
    # 检查摄像头
    cap = check_camera()
    if cap is None:
        print("错误：摄像头调用失败，请检查设备或关闭占用程序")
        return
    
    # 设置摄像头分辨率（默认硬件最大，可手动修改）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # 界面提示
    print("=" * 50)
    print("条形码识别与存储系统 V1.0")
    print("操作说明：")
    print("  - 按 's' 开始/暂停识别")
    print("  - 按 'q' 退出系统")
    print("  - 存储目录：", STORAGE_ROOT)
    print("=" * 50)
    
    recognition_enabled = True  # 是否启用识别
    last_recognition_time = 0    # 上次识别时间
    frame_count = 0              # 帧计数（用于优化显示）
    
    while is_running:
        ret, frame = cap.read()
        #frame = cv2.imread('a.jpg')
        #frame尺寸根据图片实际尺寸

        if not ret:
            logger.error(f"识别 | 无 | 失败 | 图像采集失败")
            break
        
        # 显示基础信息
        display_frame = frame.copy()
        cv2.putText(display_frame, f"recognition_enabled:{'ON' if recognition_enabled else 'OFF'}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if recognition_enabled else (0, 0, 255), 2)
        cv2.putText(display_frame, f"DIR:{STORAGE_ROOT}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
  
        # 识别逻辑（仅当启用且满足条件时）
        if recognition_enabled:
            current_time = time.time()
            # 1. 静止判定 + 频率控制
            check=is_frame_static(frame, last_frame) and (current_time - last_recognition_time) >= RECOGNITION_INTERVAL
            #logger.info(f"check:{check}")
            if is_frame_static(frame, last_frame) and (current_time - last_recognition_time) >= RECOGNITION_INTERVAL:
                last_recognition_time = current_time
                frame_count = 0
                
                # 2. 解码条形码
                #读取目录下tst.jpg图像
       
                barcodes = decode_barcode(frame)
                logger.info(f"识别 | 无 | 信息 | 识别到 {len(barcodes)} 个条形码: {[b['data'] for b in barcodes]}")
                if not barcodes:
                    cv2.putText(display_frame, "no detect", (10, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    logger.info(f"识别 | 无 | 失败 | 未识别到条形码")
                    continue
                
                # 3. 校验并标记条形码
                valid_barcodes = []
                invalid_info = []
                for barcode in barcodes:
                    data = barcode["data"]
                    pos = barcode["pos"]
                    is_valid, reason = validate_barcode(data)
                    
                    if is_valid:
                        valid_barcodes.append(barcode)
                        # 绘制绿色实线边框
                        cv2.rectangle(display_frame, (pos[0], pos[1]), 
                                     (pos[0]+pos[2], pos[1]+pos[3]), (0, 255, 0), 2)
                        cv2.putText(display_frame, "valid", (pos[0], pos[1]-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        invalid_info.append(f"{data}（{reason}）")
                        # 绘制红色虚线边框
                        cv2.rectangle(display_frame, (pos[0], pos[1]), 
                                     (pos[0]+pos[2], pos[1]+pos[3]), (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(display_frame, "invalid", (pos[0], pos[1]-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # 4. 日志记录与存储
                barcode_datas = ",".join([b["data"] for b in barcodes])
                if valid_barcodes:
                    storage_msg, success_cnt, fail_cnt = save_image(frame, valid_barcodes)
                    cv2.putText(display_frame, f"result:{storage_msg}", (10, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    logger.info(f"识别 | {barcode_datas} | 部分成功" if fail_cnt > 0 else "成功 |" 
                               f"识别{len(barcodes)}个条形码，有效{success_cnt}个，无效{len(invalid_info)}个；{storage_msg}")
                else:
                    cv2.putText(display_frame, f"All barcodes are invalid:{'; '.join(invalid_info)}", (10, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    logger.info(f"识别 | {barcode_datas} | 失败 | 所有条形码无效：{'; '.join(invalid_info)}")
        
        # 更新上一帧图像
        last_frame = frame.copy()

        
        # 获取实际摄像头分辨率
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # 按比例设置窗口大小
        window_width = 1600
        window_height = int(window_width * height / width)
        
        cv2.namedWindow("条形码识别与存储系统", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("条形码识别与存储系统", window_width, window_height)
        cv2.imshow("条形码识别与存储系统", display_frame)
        
        # 键盘事件处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            logger.info(f"系统 | 无 | 成功 | 用户退出系统")
            break
        elif key == ord("s"):
            recognition_enabled = not recognition_enabled
            status = "开启" if recognition_enabled else "关闭"
            logger.info(f"系统 | 无 | 成功 | 识别功能{status}")
            cv2.putText(display_frame, f"RECG :{status}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("条形码识别与存储系统", display_frame)
            time.sleep(0.5)
    
    # 资源释放
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"系统 | 无 | 失败 | 系统异常崩溃：{str(e)}")
        raise