# Frontend

Chat UI cho hệ thống hỏi đáp y khoa. React + Vite, không dùng thư viện UI nào.

```
npm install
npm run dev      # http://localhost:5173
```

Cần backend chạy ở `:8000` (`python serve.py` từ repo root). Vite proxy `/api` sang đó, nên
code gọi đường dẫn tương đối và không có URL nào phải cấu hình theo môi trường. Đổi đích bằng
biến `API_URL` nếu backend ở chỗ khác.

Không đặt secret vào biến `VITE_*`: Vite đóng gói chúng vào bundle. API key chỉ ở backend.

## Luồng streaming

`POST /api/v1/chat/stream` trả SSE. Hợp đồng quan trọng:

| event | Ý nghĩa |
|---|---|
| `status` | Bước đang chạy, để UI không có cảm giác treo (P95 pipeline ~30–50s) |
| `sources` | Nguồn, gửi ngay khi rerank xong — trước khi có câu trả lời |
| `token` | Mảnh text của bản nháp. **Tạm thời** |
| `done` | Câu trả lời **chốt** (sau citation_validator) + meta. UI thay toàn bộ text bằng cái này |
| `error` | Lỗi |

`citation_validator` chạy sau `generate_answer`, nên bản nháp đã stream có thể bị lọc câu hoặc
viết lại. Vì vậy UI đánh dấu text đang stream là chưa chốt ("Đang viết…", rồi "Đang kiểm tra
trích dẫn…"), và khi nhận `done` thì thay text, kèm ghi chú nếu có câu bị lược bỏ.

Dùng `fetch` + `ReadableStream` chứ không dùng `EventSource` vì endpoint là POST.

## Ghi chú môi trường

Vite ghim ở 4.5 vì máy đang chạy Node 16. Nâng Node lên 20 LTS rồi bump Vite 5 được.

## File

| File | Việc |
|---|---|
| `src/api.js` | Client SSE + health + xoá session |
| `src/App.jsx` | State hội thoại, composer, banner health |
| `src/components/Message.jsx` | Một lượt: câu trả lời, nguồn, panel |
| `src/components/Markdownish.jsx` | Render `**đậm**`, gạch đầu dòng, và `[n]` thành nhãn bấm được |
| `src/components/PipelinePanel.jsx` | Panel gập "hệ thống đã trả lời thế nào" |
| `src/styles.css` | Toàn bộ CSS, có dark mode theo `prefers-color-scheme` |
