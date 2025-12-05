import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from logger import init_logger

logger = init_logger()


def draw_text_pil(cv2_img, text, pos=(0,0), font_size=18, font_path=None,
                  font_color=(255,255,255), bg_color=(0,0,0), padding=8):
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = None
    if font_path and os.path.exists(font_path):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = None
    if font is None:
        for fn in ("msyh.ttc","msyh.ttf","simhei.ttf","simsun.ttc","arial.ttf"):
            p = os.path.join("C:\\Windows\\Fonts", fn)
            if os.path.exists(p):
                try:
                    font = ImageFont.truetype(p, font_size)
                    break
                except Exception:
                    font = None
    if font is None:
        font = ImageFont.load_default()
    x,y = pos
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        txt_w = bbox[2] - bbox[0]; txt_h = bbox[3] - bbox[1]
    except Exception:
        try:
            txt_w, txt_h = font.getsize(text)
        except Exception:
            try:
                bbox = font.getbbox(text)
                txt_w = bbox[2] - bbox[0]
                txt_h = bbox[3] - bbox[1]
            except Exception:
                txt_h = font_size; txt_w = int(len(text) * font_size * 0.6)
    if bg_color is not None:
        draw.rectangle([x - padding, y - padding, x + txt_w + padding, y + txt_h + padding], fill=bg_color)
    draw.text((x, y), text, font=font, fill=font_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_persistent_items(display_frame, items):
    for it in items:
        x,y,w,h = it["pos"]
        color = (0,200,0) if it["valid"] else (0,80,255)
        cv2.rectangle(display_frame, (x,y), (x+w, y+h), color, 4)
        label = it["data"]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        bg_x0 = x; bg_y0 = max(0, y - th - 12); bg_x1 = x + tw + 12; bg_y1 = y
        cv2.rectangle(display_frame, (bg_x0, bg_y0), (bg_x1, bg_y1), color, -1)
        display_frame = draw_text_pil(display_frame, label, pos=(bg_x0 + 6, bg_y1 - th - 2), font_size=16,
                                      font_color=(255,255,255), bg_color=None, padding=4)
    return display_frame
