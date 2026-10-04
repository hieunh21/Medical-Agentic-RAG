"""Nhóm O — ngoài phạm vi / không có thông tin (viết tay + kiểm chứng bằng code).

expected_behavior:
  decline_out_of_scope — không liên quan y khoa; phải từ chối (route out_of_scope).
  no_info              — câu y khoa nhưng YouMed không có bài; phải nói "chưa có thông tin".
  either               — liền kề y khoa nhưng corpus không trả lời được; từ chối hoặc no_info đều đúng.

Độ khó = mức khó phát hiện là "không có thông tin":
  easy   — không dính gì tới y khoa.
  medium — câu y khoa thật, bệnh hiếm không có trong corpus (build_testset.py kiểm tra thực thể
           KHÔNG xuất hiện ở bất kỳ title / alias / nội dung chunk nào; có thì loại).
  hard   — nghe như y khoa (bệnh viện, giá thuốc, kết quả xét nghiệm cá nhân...) nhưng corpus
           kiến thức không thể trả lời.
"""
from __future__ import annotations

EASY: list[str] = [
    "Giá vàng hôm nay bao nhiêu một lượng?",
    "Công thức tính diện tích hình tròn là gì?",
    "Đội tuyển Việt Nam đá vòng loại World Cup khi nào?",
    "Cách nấu phở bò ngon chuẩn vị Hà Nội?",
    "Lãi suất gửi tiết kiệm ngân hàng hiện nay là bao nhiêu?",
    "Làm sao để học tiếng Anh nhanh?",
    "Thủ đô của nước Pháp là thành phố nào?",
    "Cách đổi mật khẩu tài khoản ngân hàng điện tử?",
]

# (các cách viết của thực thể — TẤT CẢ phải có 0 lần khớp (theo ranh giới từ, đã bỏ dấu) trong
# toàn bộ title / alias / nội dung chunk, câu hỏi). Dư ra so với số cần lấy vì sẽ bị lọc: corpus
# rộng hơn dự đoán (vd. Huntington, Lyme, Sjögren, Tourette đều có bài riêng), và bệnh chỉ được
# nhắc thoáng trong bài khác (Whipple, Behçet, Kawasaki) cũng bị loại vì đáp án "chưa có thông tin" mơ hồ.
MEDIUM_CANDIDATES: list[tuple[tuple[str, ...], str]] = [
    (("pompe",), "Bệnh Pompe điều trị bằng cách nào?"),
    (("fabry",), "Bệnh Fabry là gì?"),
    (("to đầu chi", "acromegal"), "Bệnh to đầu chi do nguyên nhân nào?"),
    (("gaucher",), "Bệnh Gaucher ảnh hưởng đến cơ thể như thế nào?"),
    (("niemann",), "Bệnh Niemann-Pick có chữa được không?"),
    (("tay-sachs", "tay sachs"), "Bệnh Tay-Sachs di truyền như thế nào?"),
    (("moyamoya",), "Bệnh Moyamoya gây đột quỵ ở trẻ em như thế nào?"),
    (("angelman",), "Hội chứng Angelman là gì?"),
    (("prader",), "Hội chứng Prader-Willi biểu hiện ra sao?"),
    (("ehlers",), "Hội chứng Ehlers-Danlos ảnh hưởng đến khớp và da thế nào?"),
    (("sarcoid",), "Bệnh sarcoidosis là gì?"),
    (("amyloid",), "Bệnh amyloidosis điều trị thế nào?"),
    (("chagas",), "Bệnh Chagas lây truyền qua đường nào?"),
    (("leishmania",), "Bệnh leishmania gây ra những triệu chứng gì?"),
    (("hanta",), "Virus Hanta lây sang người như thế nào?"),
    (("buruli",), "Bệnh loét Buruli là gì?"),
    (("zollinger",), "Hội chứng Zollinger-Ellison là gì?"),
    (("waldenstr",), "Bệnh Waldenström là bệnh gì?"),
    (("rett",), "Hội chứng Rett có biểu hiện gì ở bé gái?"),
    (("pemphigus",), "Bệnh pemphigus biểu hiện thế nào?"),
]
MEDIUM_TARGET = 8

HARD: list[str] = [
    "Bệnh viện nào ở TP.HCM chữa tiểu đường tốt nhất?",
    "Giá thuốc Panadol hiện nay là bao nhiêu?",
    "Bác sĩ nào giỏi nhất về tim mạch ở Hà Nội?",
    "Nên mua bảo hiểm y tế của công ty nào?",
    "Kết quả xét nghiệm máu của tôi hôm qua có bình thường không?",
    "Thuốc của hãng A có tốt hơn thuốc của hãng B không?",
]
