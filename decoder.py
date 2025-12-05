import cv2
import numpy as np
import time
import pyzbar.pyzbar as pyzbar
from logger import init_logger
from config import FRAME_VALIDATE_COUNT

logger = init_logger()
barcode_cache = []


def update_barcode_cache(barcodes, ttl=2.0):
    global barcode_cache
    now = time.time()
    current_datas = [b["data"] for b in barcodes]
    for data in current_datas:
        found = False
        for item in barcode_cache:
            if item["data"] == data:
                item["count"] += 1
                item["timestamp"] = now
                found = True
                break
        if not found:
            barcode_cache.append({"data": data, "count": 1, "timestamp": now})
    barcode_cache = [it for it in barcode_cache if now - it["timestamp"] <= ttl]
    confirmed = []
    for item in barcode_cache:
        if item["count"] >= FRAME_VALIDATE_COUNT:
            confirmed.append(item["data"])
            item["count"] = 0
    return [b for b in barcodes if b["data"] in confirmed]


# 将原 main.py 中的 decode_barcode 函数移植到此处
def decode_barcode(frame):
    """解码图像中的CODE128条形码（ROI定位 + 多角度多尺度）"""
    results = []
    processed_frames = []
    
    # 辅助：ROI 定位（基于水平梯度 + 形态学） 
    def locate_barcode_regions(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        # Scharr/Sobel 获取水平响应（强化水平条纹）
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_x = cv2.convertScaleAbs(grad_x)
        # 平滑然后二值
        blur = cv2.GaussianBlur(grad_x, (9, 9), 0)
        _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 形态学：闭运算放大水平连通区域，适配CODE128
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
        # 再次腐蚀/膨胀处理噪声
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []
        h, w = gray.shape[:2]
        for cnt in contours:
            x,y,ww,hh = cv2.boundingRect(cnt)
            # 过滤很小或纵向过高的区域，保留宽比大的候选（典型条码形状）
            if ww < 50 or ww < hh * 1.5:
                continue
            # 限制为画面可视范围内
            x0 = max(0, x-5); y0 = max(0, y-5); x1 = min(w, x+ww+5); y1 = min(h, y+hh+5)
            rois.append((x0,y0,x1-x0,y1-y0))
        # 如果没有找到区域，返回整图作为候选
        if not rois:
            return [(0,0,w,h)]
        return rois

    # 基础预处理组合（保持你原先有效的几步）
    def base_prep(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        # 双边滤波保边去噪
        den = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        den = clahe.apply(den)
        # 锐化
        kernel = np.array([[0, -1, 0],[-1, 5,-1],[0,-1,0]])
        sharp = cv2.filter2D(den, -1, kernel)
        return sharp

    # 多角度尝试解码函数
    def try_decode(img):
        found = []
        try:
            found = pyzbar.decode(img, symbols=[pyzbar.ZBarSymbol.CODE128])
        except Exception as e:
            logger.debug(f"pyzbar.decode 异常：{e}")
        return found

    # 首先在整图做快速尝试（低成本）
    try:
        pf = base_prep(frame)
        processed_frames.append(pf)
        # 也把原灰度/otsu 加入
        _, otsu = cv2.threshold(pf, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_frames.append(otsu)
    except Exception as e:
        logger.debug(f"基础预处理异常：{e}")

    # ROI 定位并对每个 ROI 做多尺度/多角度处理
    rois = locate_barcode_regions(frame)
    for (x,y,w,h) in rois:
        try:
            roi = frame[y:y+h, x:x+w]
            prep = base_prep(roi)
            processed_frames.append(prep)

              # ------------------ 新增：透视校正（尝试检测四边形并矫正） ------------------
            def four_point_transform(image, pts):
                # 参考 OpenCV 常见四点透视变换实现
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                (tl, tr, br, bl) = rect
                widthA = np.linalg.norm(br - bl)
                widthB = np.linalg.norm(tr - tl)
                maxWidth = max(int(widthA), int(widthB))
                heightA = np.linalg.norm(tr - br)
                heightB = np.linalg.norm(tl - bl)
                maxHeight = max(int(heightA), int(heightB))
                dst = np.array([[0, 0],
                                [maxWidth - 1, 0],
                                [maxWidth - 1, maxHeight - 1],
                                [0, maxHeight - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                return warped
            # 尝试在 roi 上检测边缘轮廓，寻找近似四边形
            try:
                g = cv2.GaussianBlur(prep, (5,5), 0)
                edges = cv2.Canny(g, 30, 150)
                cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:6]
                quad_found = False
                for c in cnts:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                    if len(approx) == 4 and cv2.contourArea(approx) > 0.2 * (w*h):
                        pts = approx.reshape(4,2).astype("float32")
                        warped = four_point_transform(roi, pts)
                        # 转灰度并预处理后加入待解码集合
                        warped_prep = base_prep(warped)
                        processed_frames.append(warped_prep)
                        # 放大以提高小模块识别
                        processed_frames.append(cv2.resize(warped_prep, (int(warped_prep.shape[1]*1.8), int(warped_prep.shape[0]*1.8)), interpolation=cv2.INTER_CUBIC))
                        quad_found = True
                        break
            except Exception as e:
                logger.debug(f"透视校正尝试异常：{e}")

            # processed_frames.append(cv2.resize(prep, (int(w*1.4), int(h*1.4)), interpolation=cv2.INTER_CUBIC))
            # 加入更大放大尺度以提升小条码识别（2.2x、2.5x）
            processed_frames.append(cv2.resize(prep, (int(w*1.4), int(h*1.4)), interpolation=cv2.INTER_CUBIC))
            processed_frames.append(cv2.resize(prep, (int(w*2.2), int(h*2.2)), interpolation=cv2.INTER_CUBIC))
            processed_frames.append(cv2.resize(prep, (int(w*2.5), int(h*2.5)), interpolation=cv2.INTER_CUBIC))
            # 旋转小角度尝试（±3°, ±6°）
            for ang in (-6, -3, 3, 6):
                (rh, rw) = prep.shape[:2]
                M = cv2.getRotationMatrix2D((rw//2, rh//2), ang, 1.0)
                rot = cv2.warpAffine(prep, M, (rw, rh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
                processed_frames.append(rot)
              # ------------------ 新增：自适应二值化变体，增强对不均匀光照的鲁棒性 ------------------
            try:
                adapt = cv2.adaptiveThreshold(prep, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 31, 9)
                processed_frames.append(adapt)
                # 反色也试一次（条码黑白反转场景）
                processed_frames.append(cv2.bitwise_not(adapt))
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"ROI 预处理异常：{e}")

    # 去重处理处理帧并逐一尝试解码
    tried = set()
    all_barcodes = []
    for i, img in enumerate(processed_frames):
        key = (img.shape, img.tobytes()[:64]) if hasattr(img, 'tobytes') else (img.shape, i)
        if key in tried:
            continue
        tried.add(key)
        barcodes = try_decode(img)
        if barcodes:
            all_barcodes.extend(barcodes)
            logger.debug(f"预处理方式 {i} 识别到 {len(barcodes)} 个条形码")

    # 结果解析与去重（保留CODE128）
    for barcode in all_barcodes:
        try:
            if barcode.type != "CODE128":
                continue
            bdata = barcode.data.decode("utf-8").strip()
            if not bdata:
                continue
            (x,y,w,h) = barcode.rect
            if not any(r["data"]==bdata and abs(r["pos"][0]-x)<25 and abs(r["pos"][1]-y)<25 for r in results):
                results.append({"data":bdata, "pos":(x,y,w,h), "type":"CODE128"})
        except Exception as e:
            logger.warning(f"CODE128 解析异常：{str(e)}")

    logger.debug(f"识别 | 无 | 信息 | CODE128识别数量：{len(results)}")
    return results
