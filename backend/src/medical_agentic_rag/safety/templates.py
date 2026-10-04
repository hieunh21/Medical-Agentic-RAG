"""Template phản hồi cố định cho emergency / self_harm — KHÔNG để LLM sinh (mục 10.1).

LƯU Ý: nội dung dưới đây là bản nháp, cần người có chuyên môn (y tế / sức khoẻ tâm thần)
đọc duyệt trước khi dùng thật. Hotline chỉ lấy từ resources.yaml (số đã xác minh).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

RESOURCES_PATH = Path(__file__).parent / "resources.yaml"


@lru_cache(maxsize=1)
def _resources() -> dict:
    return yaml.safe_load(RESOURCES_PATH.read_text(encoding="utf-8")) or {}


def emergency_response() -> str:
    number = _resources().get("emergency_number", "115")
    return (
        f"Những dấu hiệu bạn mô tả có thể là tình trạng cấp cứu. Hãy gọi cấp cứu {number} "
        "hoặc đến cơ sở y tế gần nhất ngay lập tức. Nếu có người ở bên cạnh, hãy nhờ họ đi cùng "
        "và đừng tự điều trị tại nhà. Hệ thống này chỉ cung cấp thông tin tham khảo, không thay "
        "thế đánh giá của nhân viên y tế."
    )


def self_harm_response() -> str:
    res = _resources()
    number = res.get("emergency_number", "115")
    text = (
        "Mình rất tiếc khi bạn đang cảm thấy như vậy, và cảm ơn bạn đã nói ra. Bạn không phải "
        "đối mặt với điều này một mình. Hãy liên hệ ngay với một người thân hoặc bạn bè mà bạn tin "
        "tưởng và ở bên họ, hoặc trò chuyện với chuyên gia sức khoẻ tâm thần. "
        f"Nếu bạn đang gặp nguy hiểm ngay lúc này, hãy gọi cấp cứu {number} hoặc đến cơ sở y tế gần nhất."
    )
    lines = []
    for h in res.get("hotlines") or []:
        hours = f" ({h['hours']})" if h.get("hours") else ""
        lines.append(f"- {h['name']}: {h['phone']}{hours}")
    if lines:
        text += "\n\nĐường dây hỗ trợ:\n" + "\n".join(lines)
    return text


HIGH_RISK_PREFIX = (
    "Lưu ý: triệu chứng bạn nêu có thể nguy hiểm, bạn nên đi khám sớm. Nếu xuất hiện dấu hiệu nặng "
    "lên hoặc đột ngột (khó thở, đau ngực, ngất, yếu liệt, lơ mơ...), hãy gọi cấp cứu 115 hoặc đến "
    "cơ sở y tế gần nhất ngay.\n\n"
)

# Chèn vào đầu câu trả lời khi câu hỏi nhắc tới chủ đề cấp cứu nhưng là hỏi kiến thức chung (không dừng
# pipeline): người đang gặp tình huống thật mà diễn đạt chung chung vẫn thấy khuyến cáo.
EMERGENCY_TOPIC_NOTE = (
    "Lưu ý: nếu bạn hoặc người thân đang có các dấu hiệu này ngay lúc này, hãy gọi cấp cứu 115 "
    "hoặc đến cơ sở y tế gần nhất.\n\n"
)
