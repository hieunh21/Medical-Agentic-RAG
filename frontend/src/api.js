// Client cho API backend. Dùng fetch + ReadableStream chứ không dùng EventSource:
// EventSource chỉ gửi được GET, còn /api/v1/chat/stream là POST (câu hỏi nằm trong body).

/**
 * Gửi câu hỏi và nhận SSE. Gọi lại handlers theo từng sự kiện:
 *   onStatus({step, label})  — bước đang chạy
 *   onSources([...])         — nguồn, tới trước câu trả lời
 *   onToken(text)            — mảnh text của bản nháp (tạm)
 *   onDone({answer, sources, meta, session_id}) — bản CHỐT, thay toàn bộ text
 *   onError(message)
 * Trả về hàm abort().
 */
export function streamChat({ message, sessionId, signal, onStatus, onSources, onToken, onDone, onError }) {
  const controller = new AbortController();
  if (signal) signal.addEventListener("abort", () => controller.abort());

  (async () => {
    try {
      const res = await fetch("/api/v1/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, session_id: sessionId ?? null }),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) {
        throw new Error(`Máy chủ trả về HTTP ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      // SSE: các khối cách nhau bằng dòng trống; một khối có "event:" và "data:".
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let sep;
        while ((sep = buffer.indexOf("\n\n")) !== -1) {
          const block = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);

          let event = "message";
          const dataLines = [];
          for (const line of block.split("\n")) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
          }
          if (!dataLines.length) continue;

          let payload;
          try {
            payload = JSON.parse(dataLines.join("\n"));
          } catch {
            continue; // khối chưa trọn vẹn hoặc không phải JSON — bỏ qua
          }

          if (event === "status") onStatus?.(payload);
          else if (event === "sources") onSources?.(payload.sources || []);
          else if (event === "token") onToken?.(payload.text || "");
          else if (event === "done") onDone?.(payload);
          else if (event === "error") onError?.(payload.message || "Lỗi không rõ");
        }
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        onError?.(
          err.message?.includes("fetch")
            ? "Không kết nối được máy chủ. Kiểm tra backend đang chạy ở cổng 8000."
            : err.message || "Lỗi không rõ"
        );
      }
    }
  })();

  return () => controller.abort();
}

export async function fetchHealth() {
  const res = await fetch("/api/v1/health");
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function deleteSession(sessionId) {
  await fetch(`/api/v1/sessions/${sessionId}`, { method: "DELETE" });
}
