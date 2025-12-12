import os
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
load_dotenv()

# 识别参数
RECOGNITION_INTERVAL = float(os.getenv("RECOGNITION_INTERVAL", 0))
STATIC_THRESHOLD = float(os.getenv("STATIC_THRESHOLD", 90.0)) / 100  # 转为小数
SENSITIVITY = os.getenv("RECOGNITION_SENSITIVITY", "high")

# 数据格式
ORDER_PREFIX = os.getenv("ORDER_PREFIX", "")
ORDER_NUM_LENGTH = int(os.getenv("ORDER_NUM_LENGTH", 7))
SERIAL_NUM_LENGTH = int(os.getenv("SERIAL_NUM_LENGTH", 3))
SEPARATOR = os.getenv("SEPARATOR", "-")

# 存储配置
PROJECT_ROOT = Path(__file__).resolve().parent
STORAGE_ROOT = str(PROJECT_ROOT / "pic")
IMAGE_QUALITY = int(os.getenv("IMAGE_QUALITY", 85))

# 新增：测试模式配置（可在 .env 中设置 TEST_MODE=1 或 TEST_MODE=true，并可指定 TST_DIR）
TEST_MODE = os.getenv("TEST_MODE", "0").lower() in ("1", "true", "yes")
TST_DIR = os.getenv("TST_DIR", str(PROJECT_ROOT / "tst"))
# 确保目录存在（避免运行时报错）
os.makedirs(STORAGE_ROOT, exist_ok=True)
if TEST_MODE:
    os.makedirs(TST_DIR, exist_ok=True)

# 多帧验证与显示保持
FRAME_VALIDATE_COUNT = int(os.getenv("FRAME_VALIDATE_COUNT", 1))
DISPLAY_HOLD_SECONDS = float(os.getenv("DISPLAY_HOLD_SECONDS", 2.0))

# 日志目录
LOG_DIR = os.path.join(STORAGE_ROOT, "Log")
