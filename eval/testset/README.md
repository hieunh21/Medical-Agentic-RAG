# Bộ test đánh giá

Các file `*_full.jsonl` nằm trong `.gitignore` (có cụm từ trích từ YouMed, không phát hành lại).
Dựng lại bằng `python -m eval.build_testset {r|s|o|stats}`.

Nhóm O sinh từ `eval/cases_oos.py` (có commit). Nhóm S sinh từ `eval/cases_safety.py`, file này
**không commit** vì chứa các cách diễn đạt ý định tự hại dùng làm input kiểm thử — dựng lại bằng:

```
cp eval/cases_safety.example.py eval/cases_safety.py
```

rồi viết thêm ca cho đủ 6 nhãn theo hướng dẫn trong chính file đó.

| File | Nhóm | Nguồn | Gold | Dùng cho |
|---|---|---|---|---|
| `h_full.jsonl` | H — văn nói | viết tay | `gold_urls`, `must_mention`, `expected_route` | retrieval, routing, E2E |
| `m_full.jsonl` | M — so sánh / nhiều khía cạnh | viết tay | `gold_urls`, `subtype` | routing, subquery |
| `c_complex_full.jsonl` | complex | viết tay | `gold_urls` | research agent |
| `r_full.jsonl` | R — retrieval | LLM sinh từ chunk, lọc bằng code + LLM | `gold_chunk_id`, `gold_urls` | retrieval (Recall/MRR) |
| `s_full.jsonl` | S — safety | viết tay | `expected_label`, `expected_action` | safety, high-risk recall |
| `o_full.jsonl` | O — ngoài phạm vi | viết tay + kiểm chứng corpus | `expected_behavior` | no-info precision |

Chưa có: nhóm C-hội thoại (chuỗi 2–3 lượt) theo mục 5 thiết kế.
Nhóm R do LLM sinh nên dễ thiên vị: báo cáo nhóm H (viết tay) riêng và coi đó là số chính.

## Thang độ khó (`difficulty`: easy | medium | hard)

Cùng ba nhãn cho mọi nhóm, nhưng định nghĩa theo nhóm — độ khó là *khó với hệ thống*, không phải khó với người.

**R** (đo retrieval)
- `easy`: hỏi trực tiếp, nêu tên bệnh / thuật ngữ đúng như trong bài, đủ dấu.
- `medium`: diễn đạt lại, dùng tên gọi khác / thông thường; vẫn nêu rõ bệnh nào.
- `hard`: văn nói của người bệnh, mô tả tình huống hoặc triệu chứng, hạn chế nêu tên bệnh; một nửa số câu bị bỏ dấu bằng code.

Mỗi ô (section_type × mức) có số câu bằng nhau (mặc định 9) nên kết quả không bị lệch về một loại mục.
Mỗi chunk chỉ dùng cho một mức, mỗi bài tối đa 2 chunk.

**S** (đo khả năng bắt nhãn safety)
- `easy`: nêu thẳng cụm trong `safety/keywords_vi.yaml`, hoặc câu bình thường rõ ràng.
- `medium`: diễn đạt khác; lớp 1 (rule) hụt, cần lớp 2 (LLM).
- `hard`: không dấu / văn nói / gián tiếp, hoặc hard negative (câu hỏi thông tin có chứa từ khoá như "co giật", "khó thở").

**O** (đo khả năng nói "không có thông tin")
- `easy`: không liên quan y khoa → `decline_out_of_scope`.
- `medium`: bệnh hiếm có **0 lần xuất hiện** trong toàn corpus (kiểm bằng code) → `no_info`.
- `hard`: nghe như y khoa nhưng corpus kiến thức không trả lời được (bệnh viện, giá thuốc, kết quả xét nghiệm cá nhân) → `either`.

## Kiểm chứng thang độ khó

Nhãn khó/dễ chỉ có nghĩa nếu số liệu khớp. Hai kiểm tra:
1. `python -m eval.build_testset stats` in `lexical_overlap` trung bình theo mức của nhóm R (độ trùng từ giữa câu hỏi và chunk gold); kỳ vọng giảm dần easy > medium > hard.
2. Article@5 / Recall@10 của baseline retrieval chạy theo từng mức cũng phải giảm dần. Nếu không, mức đó đang gắn nhãn sai và cần sửa prompt sinh câu hỏi, không phải sửa số liệu.

## Chia dev / test

40/60 cố định, phân tầng theo (nhóm con, độ khó), suy ra từ id nên không đổi giữa các lần dựng.
Chỉ tune ngưỡng và prompt trên `dev`; báo cáo số cuối trên `test`.
Khi sửa bộ dữ liệu thì chạy lại mọi cấu hình trong bảng ablation.

## Bảng ablation

`python -m eval.run_ablation` chạy 6 cấu hình tích luỹ (`graph/build.py::Features`) trên split `test`
và ghi `data/ablation/report.md` (kết quả từng câu cache ở `data/ablation/<hàng>.jsonl`, chạy lại sẽ tiếp tục).
Metric so với gold (Article@5, chunk-hit, must_mention, nhãn safety, từ chối) là khách quan.
Faithfulness và citation precision dùng LLM-judge, chưa hiệu chuẩn bằng người — chỉ số tham khảo;
mặc định judge cùng model với bên sinh câu trả lời, đặt `EVAL_JUDGE_MODEL` để dùng model khác.
Hàng "+ P5: citation validator" có faithfulness cao một phần vì validator đã loại câu `unsupported` trước khi chấm.
