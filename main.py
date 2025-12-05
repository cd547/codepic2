"""
main.py - 精简主流程，调用模块化实现
此文件现在只负责进程控制与 UI，功能细分到 camera.py, decoder.py, display.py, storage.py, config.py, logger.py
"""
import time
import cv2
from logger import init_logger
from camera import check_camera, is_frame_static
from decoder import decode_barcode, update_barcode_cache
from display import draw_text_pil, draw_persistent_items
from storage import save_image
from config import *
import os
import glob
from pathlib import Path
logger = init_logger()

# # -------------------------- 工具函数 --------------------------
# def check_camera():
#     """检查摄像头是否可用"""
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         # 重试1次（共2次）
#         time.sleep(1)
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             logger.error(f"识别 | 无 | 失败 | 摄像头调用失败：未检测到可用摄像头或被占用")
#             return None
    
#     # 关键优化：开启自动对焦（部分摄像头支持）
#     cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 1=开启自动对焦
#     # 优化曝光（避免过暗/过亮）
#     cap.set(cv2.CAP_PROP_EXPOSURE, -4)  # -4=自动曝光（可根据实际调整，范围-10~10）
#     # 优化白平衡
#     cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # 1=自动白平衡
#     # 提高采集帧率（减少模糊）
#     cap.set(cv2.CAP_PROP_FPS, 30)
#     logger.info(f"识别 | 无 | 成功 | 摄像头调用成功")
#     return cap

# def is_frame_static(current_frame, last_frame):
#     """判定图像是否静止（像素变化率低于阈值）"""
#     if last_frame is None:
#         return False
#     # 转为灰度图计算差异
#     gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
#     gray_last = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
#     # 计算差异像素数
#     diff = cv2.absdiff(gray_current, gray_last)
#     non_zero_count = np.count_nonzero(diff)
#     total_pixels = diff.shape[0] * diff.shape[1]
#     change_rate = non_zero_count / total_pixels
#     return change_rate <= STATIC_THRESHOLD


# # ...existing code...
# def decode_barcode(frame):
#     """解码图像中的CODE128条形码（ROI定位 + 多角度多尺度）"""
#     results = []
#     processed_frames = []
    
#     # 辅助：ROI 定位（基于水平梯度 + 形态学） 
#     def locate_barcode_regions(img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
#         # Scharr/Sobel 获取水平响应（强化水平条纹）
#         grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
#         grad_x = cv2.convertScaleAbs(grad_x)
#         # 平滑然后二值
#         blur = cv2.GaussianBlur(grad_x, (9, 9), 0)
#         _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         # 形态学：闭运算放大水平连通区域，适配CODE128
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
#         closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
#         # 再次腐蚀/膨胀处理噪声
#         closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        
#         contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         rois = []
#         h, w = gray.shape[:2]
#         for cnt in contours:
#             x,y,ww,hh = cv2.boundingRect(cnt)
#             # 过滤很小或纵向过高的区域，保留宽比大的候选（典型条码形状）
#             if ww < 50 or ww < hh * 1.5:
#                 continue
#             # 限制为画面可视范围内
#             x0 = max(0, x-5); y0 = max(0, y-5); x1 = min(w, x+ww+5); y1 = min(h, y+hh+5)
#             rois.append((x0,y0,x1-x0,y1-y0))
#         # 如果没有找到区域，返回整图作为候选
#         if not rois:
#             return [(0,0,w,h)]
#         return rois

#     # 基础预处理组合（保持你原先有效的几步）
#     def base_prep(img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
#         # 双边滤波保边去噪
#         den = cv2.bilateralFilter(gray, 9, 75, 75)
#         clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
#         den = clahe.apply(den)
#         # 锐化
#         kernel = np.array([[0, -1, 0],[-1, 5,-1],[0,-1,0]])
#         sharp = cv2.filter2D(den, -1, kernel)
#         return sharp

#     # 多角度尝试解码函数
#     def try_decode(img):
#         found = []
#         try:
#             found = pyzbar.decode(img, symbols=[pyzbar.ZBarSymbol.CODE128])
#         except Exception as e:
#             logger.debug(f"pyzbar.decode 异常：{e}")
#         return found

#     # 首先在整图做快速尝试（低成本）
#     try:
#         pf = base_prep(frame)
#         processed_frames.append(pf)
#         # 也把原灰度/otsu 加入
#         _, otsu = cv2.threshold(pf, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         processed_frames.append(otsu)
#     except Exception as e:
#         logger.debug(f"基础预处理异常：{e}")

#     # ROI 定位并对每个 ROI 做多尺度/多角度处理
#     rois = locate_barcode_regions(frame)
#     for (x,y,w,h) in rois:
#         try:
#             roi = frame[y:y+h, x:x+w]
#             prep = base_prep(roi)
#             processed_frames.append(prep)

#               # ------------------ 新增：透视校正（尝试检测四边形并矫正） ------------------
#             def four_point_transform(image, pts):
#                 # 参考 OpenCV 常见四点透视变换实现
#                 rect = np.zeros((4, 2), dtype="float32")
#                 s = pts.sum(axis=1)
#                 rect[0] = pts[np.argmin(s)]
#                 rect[2] = pts[np.argmax(s)]
#                 diff = np.diff(pts, axis=1)
#                 rect[1] = pts[np.argmin(diff)]
#                 rect[3] = pts[np.argmax(diff)]
#                 (tl, tr, br, bl) = rect
#                 widthA = np.linalg.norm(br - bl)
#                 widthB = np.linalg.norm(tr - tl)
#                 maxWidth = max(int(widthA), int(widthB))
#                 heightA = np.linalg.norm(tr - br)
#                 heightB = np.linalg.norm(tl - bl)
#                 maxHeight = max(int(heightA), int(heightB))
#                 dst = np.array([[0, 0],
#                                 [maxWidth - 1, 0],
#                                 [maxWidth - 1, maxHeight - 1],
#                                 [0, maxHeight - 1]], dtype="float32")
#                 M = cv2.getPerspectiveTransform(rect, dst)
#                 warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#                 return warped
#             # 尝试在 roi 上检测边缘轮廓，寻找近似四边形
#             try:
#                 g = cv2.GaussianBlur(prep, (5,5), 0)
#                 edges = cv2.Canny(g, 30, 150)
#                 cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#                 cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:6]
#                 quad_found = False
#                 for c in cnts:
#                     peri = cv2.arcLength(c, True)
#                     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#                     if len(approx) == 4 and cv2.contourArea(approx) > 0.2 * (w*h):
#                         pts = approx.reshape(4,2).astype("float32")
#                         warped = four_point_transform(roi, pts)
#                         # 转灰度并预处理后加入待解码集合
#                         warped_prep = base_prep(warped)
#                         processed_frames.append(warped_prep)
#                         # 放大以提高小模块识别
#                         processed_frames.append(cv2.resize(warped_prep, (int(warped_prep.shape[1]*1.8), int(warped_prep.shape[0]*1.8)), interpolation=cv2.INTER_CUBIC))
#                         quad_found = True
#                         break
#             except Exception as e:
#                 logger.debug(f"透视校正尝试异常：{e}")

#             # processed_frames.append(cv2.resize(prep, (int(w*1.4), int(h*1.4)), interpolation=cv2.INTER_CUBIC))
#             # 加入更大放大尺度以提升小条码识别（2.2x、2.5x）
#             processed_frames.append(cv2.resize(prep, (int(w*1.4), int(h*1.4)), interpolation=cv2.INTER_CUBIC))
#             processed_frames.append(cv2.resize(prep, (int(w*2.2), int(h*2.2)), interpolation=cv2.INTER_CUBIC))
#             processed_frames.append(cv2.resize(prep, (int(w*2.5), int(h*2.5)), interpolation=cv2.INTER_CUBIC))
#             # 旋转小角度尝试（±3°, ±6°）
#             for ang in (-6, -3, 3, 6):
#                 (rh, rw) = prep.shape[:2]
#                 M = cv2.getRotationMatrix2D((rw//2, rh//2), ang, 1.0)
#                 rot = cv2.warpAffine(prep, M, (rw, rh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
#                 processed_frames.append(rot)
#               # ------------------ 新增：自适应二值化变体，增强对不均匀光照的鲁棒性 ------------------
#             try:
#                 adapt = cv2.adaptiveThreshold(prep, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                               cv2.THRESH_BINARY, 31, 9)
#                 processed_frames.append(adapt)
#                 # 反色也试一次（条码黑白反转场景）
#                 processed_frames.append(cv2.bitwise_not(adapt))
#             except Exception:
#                 pass
#         except Exception as e:
#             logger.debug(f"ROI 预处理异常：{e}")

#     # 去重处理处理帧并逐一尝试解码
#     tried = set()
#     all_barcodes = []
#     for i, img in enumerate(processed_frames):
#         key = (img.shape, img.tobytes()[:64]) if hasattr(img, 'tobytes') else (img.shape, i)
#         if key in tried:
#             continue
#         tried.add(key)
#         barcodes = try_decode(img)
#         if barcodes:
#             all_barcodes.extend(barcodes)
#             logger.debug(f"预处理方式 {i} 识别到 {len(barcodes)} 个条形码")

#     # 结果解析与去重（保留CODE128）
#     for barcode in all_barcodes:
#         try:
#             if barcode.type != "CODE128":
#                 continue
#             bdata = barcode.data.decode("utf-8").strip()
#             if not bdata:
#                 continue
#             (x,y,w,h) = barcode.rect
#             if not any(r["data"]==bdata and abs(r["pos"][0]-x)<25 and abs(r["pos"][1]-y)<25 for r in results):
#                 results.append({"data":bdata, "pos":(x,y,w,h), "type":"CODE128"})
#         except Exception as e:
#             logger.warning(f"CODE128 解析异常：{str(e)}")

#     logger.debug(f"识别 | 无 | 信息 | CODE128识别数量：{len(results)}")
#     return results
# # ...existing code...
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

# def save_image(frame, valid_barcodes):
#     """按规则存储图像（去重+多订单关联）"""
#     # 生成图像临时文件（避免重复存储）
#     temp_path = os.path.join(STORAGE_ROOT, f"temp_{int(time.time())}.jpg")
#     cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])
    
#     success_count = 0
#     fail_reasons = []
#     order_serials = []  # 已处理的订单号+序号组合（去重）
    
#     for barcode in valid_barcodes:
#         barcode_data = barcode["data"]
#         order_part, serial_part = barcode_data.split(SEPARATOR, 1)
#         order_num = order_part  # 完整订单号（含前缀）
#         file_name = f"{barcode_data}.jpg"
#         order_dir = os.path.join(STORAGE_ROOT, order_num)
        
#         # 跳过重复的订单号+序号组合
#         if barcode_data in order_serials:
#             continue
#         order_serials.append(barcode_data)
        
#         try:
#             # 创建订单子目录
#             os.makedirs(order_dir, exist_ok=True)
#             target_path = os.path.join(order_dir, file_name)
            
#             # 检查文件是否已存在
#             if os.path.exists(target_path):
#                 fail_reasons.append(f"{barcode_data}：文件已存在")
#                 continue
            
#             # 复制临时文件到目标路径（避免重复编码）
#             shutil.copy2(temp_path, target_path)
#             success_count += 1
#             logger.info(f"存储 | {barcode_data} | 成功 | 图像已保存至{target_path}")
#         except PermissionError:
#             fail_reasons.append(f"{barcode_data}：无写入权限")
#             logger.error(f"存储 | {barcode_data} | 失败 | 无写入权限：{order_dir}")
#         except Exception as e:
#             fail_reasons.append(f"{barcode_data}：未知错误")
#             logger.error(f"存储 | {barcode_data} | 失败 | 存储异常：{str(e)}")
    
#     # 删除临时文件
#     os.remove(temp_path)
    
#     # 返回存储结果
#     if success_count == len(valid_barcodes):
#         return "全部成功", success_count, 0
#     elif success_count > 0:
#         return f"部分成功（失败原因：{'; '.join(fail_reasons)}）", success_count, len(fail_reasons)
#     else:
#         return f"全部失败（原因：{'; '.join(fail_reasons)}）", 0, len(fail_reasons)

# # ...existing code...
# # 新增：多帧确认缓存更新函数
# def update_barcode_cache(barcodes, ttl=2.0):
#     """更新全局 barcode_cache 并返回已确认（连续出现达到 FRAME_VALIDATE_COUNT）的条码对象列表"""
#     global barcode_cache
#     now = time.time()
#     current_datas = [b["data"] for b in barcodes]
#     # 增量记录当前帧出现的条码
#     for data in current_datas:
#         found = False
#         for item in barcode_cache:
#             if item["data"] == data:
#                 item["count"] += 1
#                 item["timestamp"] = now
#                 found = True
#                 break
#         if not found:
#             barcode_cache.append({"data": data, "count": 1, "timestamp": now, "info": {}})
#     # 移除过期项
#     barcode_cache = [it for it in barcode_cache if now - it["timestamp"] <= ttl]
#     # 收集已确认的条码（达到阈值）
#     confirmed = []
#     for item in barcode_cache:
#         if item["count"] >= FRAME_VALIDATE_COUNT:
#             confirmed.append(item["data"])
#             item["count"] = 0  # 避免重复触发
#     # 返回与原 barcode 对象对应的列表
#     return [b for b in barcodes if b["data"] in confirmed]


# def draw_text_pil(cv2_img, text, pos=(0,0), font_size=18, font_path=None,
#                   font_color=(255,255,255), bg_color=(0,0,0), padding=8):
#     """在 cv2 (BGR) 图像上使用 PIL 绘制支持中文的文字和背景，返回修改后的 BGR 图像"""
#     # 转为 PIL RGB
#     pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(pil_img)
#     # 尝试加载常见中文字体（Windows 系统）
#     font = None
#     if font_path and os.path.exists(font_path):
#         try:
#             font = ImageFont.truetype(font_path, font_size)
#         except Exception:
#             font = None
#     if font is None:
#         for fn in ("msyh.ttc", "msyh.ttf", "simhei.ttf", "simsun.ttc", "arial.ttf"):
#             p = os.path.join("C:\\Windows\\Fonts", fn)
#             if os.path.exists(p):
#                 try:
#                     font = ImageFont.truetype(p, font_size)
#                     break
#                 except Exception:
#                     font = None
#     if font is None:
#         font = ImageFont.load_default()

#     x, y = pos

#     # 兼容多版本 Pillow 的文本尺寸计算
#     try:
#         # Pillow >= 8: textbbox 可用，返回 (left, top, right, bottom)
#         bbox = draw.textbbox((0, 0), text, font=font)
#         txt_w = bbox[2] - bbox[0]
#         txt_h = bbox[3] - bbox[1]
#     except Exception:
#         try:
#             # 常见退路：font.getsize
#             txt_w, txt_h = font.getsize(text)
#         except Exception:
#             try:
#                 # font.getbbox 也可能存在
#                 bbox = font.getbbox(text)
#                 txt_w = bbox[2] - bbox[0]
#                 txt_h = bbox[3] - bbox[1]
#             except Exception:
#                 # 最后兜底：按字符数估算宽度
#                 txt_h = font_size
#                 txt_w = int(len(text) * font_size * 0.6)

#     # 背景矩形
#     if bg_color is not None:
#         draw.rectangle([x - padding, y - padding, x + txt_w + padding, y + txt_h + padding], fill=bg_color)
#     # 绘文字（PIL 用 RGB 颜色）
#     draw.text((x, y), text, font=font, fill=font_color)
#     # 转回 cv2 BGR
#     return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def _collect_test_images(tst_dir):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(tst_dir, e)))
    return sorted(files)



# -------------------------- 主逻辑 --------------------------
def main():
    # cap = check_camera()
    # if cap is None:
    #     print("错误：摄像头不可用")
    #     return

    # 根据 config 中的 TEST_MODE 决定是否启用测试模式（从 TST_DIR 读取图片）
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
        cap = check_camera()
        if cap is None:
            print("错误：摄像头不可用")
            return

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
        if recognition_enabled and (TEST_MODE or is_frame_static(frame, last_frame)) and (current_time - last_recognition_time) >= RECOGNITION_INTERVAL:
            last_recognition_time = current_time
            barcodes = decode_barcode(frame)
            logger.info(f"识别到 {len(barcodes)} 条码")
            if barcodes:
                confirmed = update_barcode_cache(barcodes)
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
                        save_image(frame, valid)

        if time.time() - last_displayed["time"] <= DISPLAY_HOLD_SECONDS:
            display = draw_persistent_items(display, last_displayed["items"])

        status = f"RECG:{'ON' if recognition_enabled else 'OFF'}  DIR:{STORAGE_ROOT}"
        display = draw_text_pil(display, status, pos=(12,8), font_size=18, bg_color=(0,0,0))

        cv2.imshow("条形码识别与存储系统", display)
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
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = init_logger()
        logger.error(f"系统 | 无 | 失败 | 系统异常崩溃：{str(e)}")
        raise