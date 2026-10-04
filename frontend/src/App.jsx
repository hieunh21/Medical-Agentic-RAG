import { useCallback, useEffect, useRef, useState } from "react";
import Message from "./components/Message";
import { deleteSession, fetchHealth, streamChat } from "./api";

const EXAMPLES = [
  "Bệnh Addison là gì?",
  "Triệu chứng của sốt xuất huyết?",
  "Hay bị choáng lúc đứng dậy là sao?",
  "Nguyên nhân, triệu chứng và cách phòng zona?",
];

const DISCLAIMER =
  "Thông tin tham khảo từ các bài viết y khoa trên YouMed, không thay thế chẩn đoán hoặc chỉ định của bác sĩ.";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [health, setHealth] = useState(null);
  const abortRef = useRef(null);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => setHealth({ ok: false, services: [] }));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const patchLast = useCallback((patch) => {
    setMessages((prev) => {
      const next = prev.slice();
      const i = next.length - 1;
      next[i] = typeof patch === "function" ? patch(next[i]) : { ...next[i], ...patch };
      return next;
    });
  }, []);

  const send = useCallback(
    (text) => {
      const question = text.trim();
      if (!question || busy) return;
      setInput("");
      setBusy(true);
      setMessages((prev) => [
        ...prev,
        { role: "user", content: question },
        { role: "bot", content: "", sources: [], state: "streaming", step: "", filtered: 0 },
      ]);

      abortRef.current = streamChat({
        message: question,
        sessionId,
        onStatus: ({ step, label }) =>
          patchLast((m) => ({
            ...m,
            step: label,
            // generate xong -> validator đang chạy: bản nháp trên màn hình chưa phải bản chốt
            state: step === "citation_validator" ? "validating" : m.state,
          })),
        onSources: (sources) => patchLast((m) => (m.content ? m : { ...m, sources })),
        onToken: (piece) => patchLast((m) => ({ ...m, content: m.content + piece })),
        onDone: ({ session_id, answer, sources, meta }) => {
          setSessionId(session_id);
          patchLast((m) => {
            // đếm câu bị validator loại, để nói rõ vì sao text vừa đổi
            const before = m.content.split(/(?<=[.!?])\s+/).filter((s) => s.trim()).length;
            const after = answer.split(/(?<=[.!?])\s+/).filter((s) => s.trim()).length;
            return {
              ...m,
              content: answer,
              sources: sources?.length ? sources : m.sources,
              meta,
              state: "done",
              step: "",
              filtered: m.content && after < before ? before - after : 0,
            };
          });
          setBusy(false);
        },
        onError: (message) => {
          patchLast({ error: message, state: "done" });
          setBusy(false);
        },
      });
    },
    [busy, patchLast, sessionId]
  );

  const reset = async () => {
    abortRef.current?.();
    if (sessionId) await deleteSession(sessionId).catch(() => {});
    setSessionId(null);
    setMessages([]);
    setBusy(false);
    inputRef.current?.focus();
  };

  const down = health && !health.ok;

  return (
    <div className="app">
      <header>
        <div>
          <h1>Trợ lý thông tin y khoa</h1>
          <p className="sub">Hỏi đáp dựa trên 613 bài viết YouMed · mọi ý đều dẫn nguồn</p>
        </div>
        {messages.length > 0 && (
          <button className="ghost" onClick={reset} type="button">
            Hội thoại mới
          </button>
        )}
      </header>

      {down && (
        <div className="banner">
          Một số thành phần chưa sẵn sàng:{" "}
          {health.services.filter((s) => !s.ok).map((s) => `${s.name} (${s.detail})`).join(", ")}. Cần Qdrant
          ở :6333 và model server ở :8001.
        </div>
      )}

      <main>
        {messages.length === 0 ? (
          <div className="empty">
            <p className="hint">Thử một câu hỏi:</p>
            <div className="examples">
              {EXAMPLES.map((q) => (
                <button key={q} onClick={() => send(q)} disabled={busy} type="button">
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m, i) => <Message key={i} msg={m} />)
        )}
        <div ref={bottomRef} />
      </main>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <textarea
          ref={inputRef}
          value={input}
          rows={1}
          placeholder="Hỏi về triệu chứng, bệnh lý hoặc cách chăm sóc sức khoẻ…"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send(input);
            }
          }}
          disabled={busy}
        />
        <button type="submit" disabled={busy || !input.trim()}>
          {busy ? "Đang trả lời…" : "Gửi"}
        </button>
      </form>

      <footer>{DISCLAIMER}</footer>
    </div>
  );
}
