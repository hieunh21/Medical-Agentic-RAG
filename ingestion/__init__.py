"""Nạp .env ngay khi import package — các module ở đây đọc os.getenv() lúc import (QDRANT_URL,
MODEL_SPACE...). Trước đây chỉ backend/config.py gọi load_dotenv(), nên chạy `python -m ingestion.ingest`
thì .env bị bỏ qua, và thứ tự import trong backend quyết định giá trị nào được dùng.
Biến môi trường thật (Render, CI) luôn thắng: load_dotenv không ghi đè biến đã có."""
from dotenv import load_dotenv

load_dotenv()
