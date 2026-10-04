# eval

Mỗi script trả lời một câu hỏi. Tất cả đọc bộ test ở `eval/testset/` (xem README ở đó để biết
cách dựng lại) và ghi kết quả vào `data/` (đã gitignore).

## Dựng bộ test

| Lệnh | Việc |
|---|---|
| `python -m eval.build_testset r` | Sinh nhóm R bằng LLM từ chunk, phân tầng theo section_type × độ khó |
| `python -m eval.build_testset s` | Nhóm S (safety) từ `cases_safety.py` |
| `python -m eval.build_testset o` | Nhóm O (ngoài phạm vi), kiểm chứng thực thể vắng mặt trong corpus |
| `python -m eval.build_testset stats` | Thống kê bộ test + kiểm tra thang độ khó |

## Đo hệ thống hiện tại

| Lệnh | Câu hỏi nó trả lời | Chi phí |
|---|---|---|
| `python -m eval.run_eval` | **Hệ thống đang ở mức nào?** Bảng điểm trên H + O + S | ~450 lời gọi LLM |
| `python -m eval.run_eval --retrieval` | Retrieval của hệ thống trên toàn bộ 234 câu R | ~2 lời gọi/câu |
| `python -m eval.run_retrieval --testset ... --mode ...` | dense vs sparse vs hybrid vs hybrid+rerank | 0 lời gọi LLM |
| `python -m eval.run_routing_accuracy` | `analyze_query` phân loại đúng bao nhiêu? Confusion matrix | 1 lời gọi/câu |

Thêm `--report-only` để dựng lại bảng từ cache mà không gọi LLM. Kết quả cache theo từng câu nên
chạy lại chỉ chạy câu còn thiếu.

## Thí nghiệm một lần (giữ lại làm bằng chứng cho quyết định trong code)

| Script | Quyết định nó dẫn tới |
|---|---|
| `run_routing_compare.py` | `comparison` dùng hybrid một lượt, không dùng subquery (0.972 vs 0.833 article_coverage); `multi_aspect` thì ngược lại |
| `run_section_lookup_compare.py` | `section_lookup` gộp ứng viên lọc theo bài với ứng viên không lọc (Article@5 0.59 → 0.97) |
| `run_corrective.py` | Vòng sửa lỗi của Phase 2: rescue rate vs harm rate |

## File dùng chung

- `harness.py` — chạy 1 câu qua graph, chấm trích dẫn bằng LLM-judge
- `metrics.py` — Article@5, Recall/MRR/nDCG, khoảng tin cậy Wilson, P95
- `cases_safety.py` (không commit, dựng từ `cases_safety.example.py`), `cases_oos.py` — nguồn của nhóm S và O
