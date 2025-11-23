import cv2
import pyzbar.pyzbar as pyzbar
import numpy as np
import os
import logging
from datetime import datetime

# 复用您现有的解码函数，稍作修改以适应图片处理
def decode_barcode_from_image(image_path):
    """从图片文件中解码CODE128条形码"""
    # 读取图片
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"无法读取图片: {image_path}")
        return []
    
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
        print("不支持CODE128类型识别，请更新pyzbar库")
        return results
    
    # 核心预处理解码（按效果优先级遍历，收集所有识别结果）
    all_barcodes = []
    #将processed_frames中的每个图片输出到当前目录下的tmp下
    for i, img in enumerate(processed_frames):
        cv2.imwrite(f'tmp/img_{i}.png', img)
    for i, img in enumerate(processed_frames):
        try:
            barcodes = pyzbar.decode(img, symbols=[symbol_type])  # 只解码CODE128
            if barcodes:
                all_barcodes.extend(barcodes)  # 收集所有识别结果
                print(f"使用CODE128专属预处理方式 {i} 识别到 {len(barcodes)} 个条形码")
                print(f"{i} 识别到 {barcodes} ")
        except Exception as e:
            print(f"预处理方式 {i} 解码失败：{str(e)}")
            continue
    
    # 小尺寸CODE128适配（多比例缩放+二次增强）
    if not all_barcodes:
        print("常规预处理未识别到，尝试CODE128专属缩放")
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
                    print(f"CODE128缩放 {scale} 倍后识别到 {len(barcodes)} 个条形码")
            except Exception as e:
                print(f"缩放 {scale} 倍解码异常：{str(e)}")
                continue
    
    # -------------------------- 结果解析 --------------------------
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
            print(f"CODE128解码异常（数据：{barcode.data}）：{str(e)}")
    
    return results

def validate_barcode(barcode_data):
    """校验条形码有效性（含订单号+序号）"""
    # 这里可以复用您原有的校验逻辑
    # 示例中简化处理
    return True, "有效"

def process_image_barcode(image_path):
    """处理单张图片的条形码识别"""
    print(f"正在处理图片: {image_path}")
    
    # 解码条形码
    barcodes = decode_barcode_from_image(image_path)
    
    if not barcodes:
        print("未识别到条形码")
        return
    
    print(f"识别到 {len(barcodes)} 个条形码:")
    for i, barcode in enumerate(barcodes):
        data = barcode["data"]
        pos = barcode["pos"]
        is_valid, reason = validate_barcode(data)
        
        status = "有效" if is_valid else f"无效({reason})"
        print(f"  {i+1}. {data} - {status} (位置: x={pos[0]}, y={pos[1]}, w={pos[2]}, h={pos[3]})")

# 主函数 - 处理单张图片或目录下的所有图片
def main():
    """在当前目录下查找并处理所有图片文件"""
    # 获取当前目录
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")
    
    # 支持的图片格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    # 查找当前目录下的所有图片文件
    image_files = [f for f in os.listdir(current_dir) 
                   if os.path.isfile(os.path.join(current_dir, f)) and f.lower().endswith(image_extensions)]
    
    if not image_files:
        print("当前目录中未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 处理每个图片文件
    for image_file in image_files:
        image_path = os.path.join(current_dir, image_file)
        print("\n" + "="*50)
        process_image_barcode(image_path)

if __name__ == "__main__":
    main()