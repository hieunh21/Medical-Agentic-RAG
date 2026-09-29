"""Gán section_type cho heading bằng rule (không cần LLM).

Kiểm tra theo thứ tự từ trên xuống, khớp rule đầu tiên thì dừng.
"""
from __future__ import annotations

RULES: list[tuple[str, list[str]]] = [
    ("when_to_see_doctor", ["khi nào", "đi khám", "gặp bác sĩ"]),
    ("symptom", ["triệu chứng", "dấu hiệu", "biểu hiện", "nhận biết", "làm sao biết"]),
    ("cause_risk", ["nguyên nhân", "yếu tố nguy cơ", "tăng nguy cơ", "do đâu"]),
    ("diagnosis", ["chẩn đoán", "xét nghiệm", "phân biệt"]),
    ("treatment", ["điều trị", "thuốc", "chữa"]),
    ("prevention", ["phòng ngừa", "phòng tránh", "phòng bệnh"]),
    ("complication_prognosis", ["biến chứng", "nguy hiểm", "sống bao lâu", "tiên lượng"]),
    ("care", ["chăm sóc", "chế độ ăn", "sinh hoạt"]),
    ("overview", ["là gì", "tổng quan", "thế nào là"]),
]


def classify_section_type(heading: str) -> str:
    text = heading.lower()
    for section_type, keywords in RULES:
        if any(kw in text for kw in keywords):
            return section_type
    return "other"
