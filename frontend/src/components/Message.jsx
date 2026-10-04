import { useState } from "react";
import Markdownish from "./Markdownish";
import PipelinePanel from "./PipelinePanel";

function Sources({ sources, highlight }) {
  if (!sources?.length) return null;
  return (
    <ol className="sources">
      {sources.map((s) => (
        <li key={s.n} id={`src-${s.n}`} className={highlight === s.n ? "hit" : ""}>
          <a href={s.url} target="_blank" rel="noreferrer noopener">
            {s.title}
          </a>
          {s.updated_date && <span className="updated"> · cập nhật {s.updated_date}</span>}
        </li>
      ))}
    </ol>
  );
}

export default function Message({ msg }) {
  const [open, setOpen] = useState(false);
  const [highlight, setHighlight] = useState(null);

  if (msg.role === "user") {
    return (
      <div className="msg user">
        <div className="bubble">{msg.content}</div>
      </div>
    );
  }

  const streaming = msg.state === "streaming";
  const validating = msg.state === "validating";

  return (
    <div className="msg bot">
      <div className="bubble">
        {msg.error ? (
          <p className="err">{msg.error}</p>
        ) : (
          <>
            {msg.content ? (
              <Markdownish
                text={msg.content}
                sources={msg.sources}
                onCite={(n) => {
                  setHighlight(n);
                  document.getElementById(`src-${n}`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
                  setTimeout(() => setHighlight(null), 1600);
                }}
              />
            ) : (
              <p className="steps">{msg.step || "Đang xử lý"}…</p>
            )}

            {/* Bản nháp đã stream có thể bị citation_validator lọc lại, nên nói rõ là chưa chốt */}
            {(streaming || validating) && msg.content && (
              <p className="provisional">{validating ? "Đang kiểm tra trích dẫn…" : "Đang viết…"}</p>
            )}

            {msg.filtered > 0 && (
              <p className="note">
                Đã lược bỏ {msg.filtered} câu chưa đủ nguồn sau khi kiểm tra trích dẫn.
              </p>
            )}

            {msg.sources?.length > 0 && (
              <>
                <h4 className="src-head">Nguồn</h4>
                <Sources sources={msg.sources} highlight={highlight} />
              </>
            )}

            {msg.meta && <PipelinePanel meta={msg.meta} open={open} onToggle={() => setOpen((v) => !v)} />}
          </>
        )}
      </div>
    </div>
  );
}
