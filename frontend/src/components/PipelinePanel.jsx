// Panel "hệ thống đã trả lời thế nào" — mặc định gập. Cho thấy đường đi trong graph và
// chi phí của mỗi câu: đây là phần phân biệt hệ thống này với một lần gọi LLM đơn thuần.

const ROUTE_LABELS = {
  definition: "Hỏi khái niệm",
  section_lookup: "Hỏi một khía cạnh của bệnh đã nêu tên",
  symptom_to_condition: "Mô tả triệu chứng, chưa nêu tên bệnh",
  comparison: "So sánh hai đối tượng",
  multi_aspect: "Hỏi nhiều khía cạnh cùng lúc",
  complex: "Cần nối thông tin nhiều bài",
  out_of_scope: "Ngoài phạm vi y khoa",
};

const SAFETY_LABELS = {
  normal: "bình thường",
  high_risk: "triệu chứng có thể nguy hiểm",
  medication_dosing: "hỏi liều thuốc",
  diagnosis_request: "xin chẩn đoán",
  emergency: "cấp cứu",
  self_harm: "nguy cơ tự hại",
};

function Row({ label, children }) {
  if (children === null || children === undefined || children === "") return null;
  return (
    <div className="pp-row">
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

export default function PipelinePanel({ meta, open, onToggle }) {
  if (!meta) return null;
  const claims = meta.claims || {};
  const nClaims = (claims.supported || 0) + (claims.partial || 0) + (claims.unsupported || 0);
  const evidence =
    meta.evidence_status === "sufficient" ? "đủ" : meta.evidence_status === "partial" ? "một phần" : "không đủ";

  return (
    <div className="pipeline">
      <button className="pp-toggle" onClick={onToggle} aria-expanded={open} type="button">
        <span className={`chev ${open ? "open" : ""}`} aria-hidden="true">
          ▸
        </span>
        Hệ thống đã trả lời thế nào
      </button>

      {open && (
        <dl className="pp-body">
          <Row label="Loại câu hỏi">
            {ROUTE_LABELS[meta.question_type] || meta.question_type || "—"}
          </Row>
          {meta.is_followup && meta.standalone_question && (
            <Row label="Hiểu thành câu độc lập">
              <em>{meta.standalone_question}</em>
            </Row>
          )}
          <Row label="Bằng chứng">
            {evidence}
            {meta.coverage != null && ` · coverage ${meta.coverage.toFixed(2)}`}
            {meta.n_relevant != null && ` · ${meta.n_relevant} đoạn liên quan`}
          </Row>
          <Row label="Ngữ cảnh dùng để trả lời">
            {meta.n_chunks} đoạn từ {meta.n_articles} bài
          </Row>
          {nClaims > 0 && (
            <Row label="Kiểm tra trích dẫn">
              {claims.supported || 0}/{nClaims} câu có nguồn hỗ trợ
              {claims.partial ? ` · ${claims.partial} một phần` : ""}
              {claims.unsupported ? ` · ${claims.unsupported} bị loại` : ""}
            </Row>
          )}
          {meta.corrections > 0 && <Row label="Số lần tự sửa truy vấn">{meta.corrections}</Row>}
          {meta.regenerations > 0 && <Row label="Viết lại câu trả lời">{meta.regenerations} lần</Row>}
          {meta.agent_tool_calls > 0 && <Row label="Agent gọi tool">{meta.agent_tool_calls} lần</Row>}
          {meta.safety_label && meta.safety_label !== "normal" && (
            <Row label="Nhãn an toàn">{SAFETY_LABELS[meta.safety_label] || meta.safety_label}</Row>
          )}
          <Row label="Chi phí">
            {meta.llm_calls} lời gọi LLM · {(meta.elapsed_ms / 1000).toFixed(1)}s
          </Row>
        </dl>
      )}
    </div>
  );
}
