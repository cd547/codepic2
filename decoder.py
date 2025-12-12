import cv2
import numpy as np
import time
import pyzbar.pyzbar as pyzbar
from logger import init_logger
from config import FRAME_VALIDATE_COUNT

import os
import sys
from contextlib import contextmanager
import threading
import queue as _queue

logger = init_logger()
barcode_cache = []


@contextmanager
def _suppress_stderr():
    """Temporarily redirect C/POSIX-level stderr to os.devnull to suppress zbar C assertions."""
    try:
        fd = sys.stderr.fileno()
    except Exception:
        yield
        return
    old_fd = os.dup(fd)
    try:
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), fd)
        yield
    finally:
        try:
            os.dup2(old_fd, fd)
        except Exception:
            pass
        try:
            os.close(old_fd)
        except Exception:
            pass


def update_barcode_cache(barcodes, ttl=2.0):
    print("1")
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
    print(f"barcode_cache: {barcode_cache}")
    confirmed = []
    for item in barcode_cache:
        if item["count"] >= FRAME_VALIDATE_COUNT:
            confirmed.append(item["data"])
            item["count"] = 0
    return [b for b in barcodes if b["data"] in confirmed]


def decode_barcode(frame):
    print("11111111111111")
    """更高效的解码入口：先做快速预检，再在必要时做有限数量的高级预处理与解码。"""
    results = []
    processed_frames = []

    # 快速预检：廉价梯度能量过滤，避免把无条码帧送到重流程
    def cheap_edge_check(img, edge_thresh_ratio=0.0001):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            small = cv2.resize(gray, (0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
            gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
            gx = np.abs(gx)
            strong = (gx > 30).astype(np.uint8)
            ratio = np.count_nonzero(strong) / (strong.shape[0] * strong.shape[1])
            return ratio >= edge_thresh_ratio
        except Exception:
            return True

    # 基础预处理（较轻量）
    def base_prep(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        den = cv2.GaussianBlur(gray, (5,5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        den = clahe.apply(den)
        kernel = np.array([[0, -1, 0],[-1, 5,-1],[0,-1,0]])
        sharp = cv2.filter2D(den, -1, kernel)
        return sharp

    # 受限的 try_decode（保留连续/uint8 修正与 stderr 抑制）
    def try_decode(img):
        found = []
        try:
            img = np.ascontiguousarray(img)
            if hasattr(img, 'dtype') and img.dtype != np.uint8:
                img = img.astype(np.uint8)
            preferred_symbols = [
                pyzbar.ZBarSymbol.CODE128
            ]
            with _suppress_stderr():
                try:
                    found = pyzbar.decode(img, symbols=preferred_symbols)
                    print(f"f:{found}")
                except Exception:
                    found = pyzbar.decode(img)
            if not found:
                try:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
                    found = pyzbar.decode(gray, symbols=preferred_symbols)
                except Exception:
                    pass
            if found:
                try:
                    debug_list = [(f.type, f.data.decode('utf-8', errors='ignore')) for f in found]
                except Exception:
                    debug_list = [getattr(f, 'type', None) for f in found]
                logger.info(f"pyzbar.decode 返回 {len(found)} 条码: {debug_list}")
                print(f"pyzbar.decode 返回 {len(found)} 条码: {debug_list}")
        except Exception as e:
            logger.debug(f"try_decode 异常：{e}")
        return found

    # 先做廉价预检：若无明显条纹结构则跳过重流程
    try:
        if not cheap_edge_check(frame, edge_thresh_ratio=0.0001):
            try:
                small = cv2.resize(frame, (0,0), fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)
                with _suppress_stderr():
                    # 不限制符号类型，避免缩小图像时漏掉非 CODE128 类型
                    quick = pyzbar.decode(small)
                if not quick:
                    return []
            except Exception:
                pass
    except Exception:
        pass

    # 有限的预处理与解码路径
    try:
        pf = base_prep(frame)
        processed_frames.append(pf)
        _, otsu = cv2.threshold(pf, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_frames.append(otsu)
    except Exception as e:
        logger.debug(f"基础预处理异常：{e}")

    rois = []
    try:
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray_full, cv2.CV_32F, 1, 0, ksize=3)
        grad_x = cv2.convertScaleAbs(grad_x)
        blur = cv2.GaussianBlur(grad_x, (9,9), 0)
        _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,5))
        closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray_full.shape[:2]
        for cnt in contours:
            x,y,ww,hh = cv2.boundingRect(cnt)
            if ww < 40 or ww < hh * 1.4:
                continue
            x0 = max(0, x-4); y0 = max(0, y-4); x1 = min(w, x+ww+4); y1 = min(h, y+hh+4)
            rois.append((x0,y0,x1-x0,y1-y0))
    except Exception as e:
        rois = [(0,0,frame.shape[1], frame.shape[0])]

    MAX_FRAMES = 6
    for (x,y,w,h) in rois:
        if len(processed_frames) >= MAX_FRAMES:
            break
        try:
            roi = frame[y:y+h, x:x+w]
            prep = base_prep(roi)
            processed_frames.append(prep)
            try:
                scale = 1.6 if max(w,h) < 400 else 1.4
                new_w = max(1, int(prep.shape[1]*scale))
                new_h = max(1, int(prep.shape[0]*scale))
                processed_frames.append(cv2.resize(prep, (new_w, new_h), interpolation=cv2.INTER_CUBIC))
            except Exception:
                pass
            try:
                (rh, rw) = prep.shape[:2]
                M = cv2.getRotationMatrix2D((rw//2, rh//2), 3, 1.0)
                rot = cv2.warpAffine(prep, M, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
                processed_frames.append(rot)
            except Exception:
                pass
            try:
                adapt = cv2.adaptiveThreshold(prep, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 31, 9)
                processed_frames.append(adapt)
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"ROI 预处理异常：{e}")

    tried = set()
    all_barcodes = []
    for i, img in enumerate(processed_frames):
        if len(all_barcodes) > 0:
            break
        key = (img.shape, img.tobytes()[:64]) if hasattr(img, 'tobytes') else (img.shape, i)
        if key in tried:
            continue
        tried.add(key)
        barcodes = try_decode(img)
        if barcodes:
            all_barcodes.extend(barcodes)
            logger.info(f"预处理方式 {i} 识别到 {len(barcodes)} 个条形码")
            print(f"预处理方式 {i} 识别到 {len(barcodes)} 个条形码")

    for barcode in all_barcodes:
        try:
            if barcode.type != "CODE128":
                continue
            bdata = barcode.data.decode("utf-8").strip()
            if not bdata:
                continue
            (bx,by,bw,bh) = barcode.rect
            if not any(r["data"]==bdata and abs(r["pos"][0]-bx)<25 and abs(r["pos"][1]-by)<25 for r in results):
                results.append({"data":bdata, "pos":(bx,by,bw,bh), "type":"CODE128"})
        except Exception as e:
            logger.warning(f"CODE128 解析异常：{str(e)}")

    logger.debug(f"识别 | 无 | 信息 | CODE128识别数量：{len(results)}")
    return results


# -------------------- 后台解码线程 --------------------
class DecoderWorker(threading.Thread):
    """后台解码线程：只保留最新帧进行解码，完成后把 (barcodes, frame) 放入 results_queue。"""
    def __init__(self, results_queue, min_interval=0.5, poll_timeout=0.05):
        super().__init__(daemon=True)
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.results_queue = results_queue
        self.min_interval = float(min_interval)
        self.poll_timeout = float(poll_timeout)
        self._last_decode = 0.0

    def put_frame(self, frame):
        with self._lock:
            try:
                self._latest = frame.copy()
            except Exception:
                self._latest = frame

    def run(self):
        while not self._stop.is_set():
            frame = None
            with self._lock:
                if self._latest is not None:
                    frame = self._latest
                    self._latest = None
            if frame is None:
                self._stop.wait(self.poll_timeout)
                continue
            now = time.time()
            if now - self._last_decode < self.min_interval:
                continue
            self._last_decode = now
            try:
               
                # try:
                #     small = cv2.resize(frame, (0,0), fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)
                #     with _suppress_stderr():
                #         print(f'..{small}')
                #         quick = pyzbar.decode(small, symbols=[pyzbar.ZBarSymbol.CODE128])
                #         print(f'..quick:{quick}')
                # except Exception:
                #     quick = []
                # if not quick:
                #     continue
                barcodes = decode_barcode(frame)
                try:
                    self.results_queue.put_nowait((barcodes, frame))
                except _queue.Full:
                    pass
            except Exception as e:
                logger.debug(f"DecoderWorker 异常：{e}")

    def stop(self):
        self._stop.set()

# -------------------- end DecoderWorker --------------------
