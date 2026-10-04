// Render câu trả lời: **đậm**, gạch đầu dòng, và biến [n] thành nhãn trích dẫn bấm được.
// Tự viết thay vì thêm thư viện markdown: câu trả lời chỉ dùng vài dạng, và quan trọng hơn là
// phải kiểm soát được cách [n] biến thành link tới nguồn.

const BOLD = /\*\*(.+?)\*\*/g;
const REF = /\[(\d+)\]/g;

function withRefs(text, sources, onCite, keyPrefix) {
  const out = [];
  let last = 0;
  let m;
  REF.lastIndex = 0;
  while ((m = REF.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const n = Number(m[1]);
    const src = sources?.[n - 1];
    out.push(
      <button
        key={`${keyPrefix}-ref-${m.index}`}
        className="cite"
        title={src ? `${src.title}\n${src.url}` : `Nguồn ${n}`}
        onClick={() => onCite?.(n)}
        type="button"
      >
        {n}
      </button>
    );
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function inline(text, sources, onCite, keyPrefix) {
  // tách **đậm** trước, phần còn lại xử lý [n]
  const nodes = [];
  let last = 0;
  let m;
  BOLD.lastIndex = 0;
  while ((m = BOLD.exec(text)) !== null) {
    if (m.index > last) nodes.push(...withRefs(text.slice(last, m.index), sources, onCite, `${keyPrefix}-${last}`));
    nodes.push(<strong key={`${keyPrefix}-b-${m.index}`}>{withRefs(m[1], sources, onCite, `${keyPrefix}-b${m.index}`)}</strong>);
    last = m.index + m[0].length;
  }
  if (last < text.length) nodes.push(...withRefs(text.slice(last), sources, onCite, `${keyPrefix}-${last}`));
  return nodes;
}

export default function Markdownish({ text, sources, onCite }) {
  const lines = (text || "").split("\n");
  const blocks = [];
  let list = null;

  const flush = () => {
    if (list) {
      blocks.push(<ul key={`ul-${blocks.length}`}>{list}</ul>);
      list = null;
    }
  };

  lines.forEach((raw, i) => {
    const line = raw.trim();
    if (!line) {
      flush();
      return;
    }
    const bullet = line.match(/^[-*•]\s+(.*)$/);
    if (bullet) {
      list = list || [];
      list.push(<li key={`li-${i}`}>{inline(bullet[1], sources, onCite, `l${i}`)}</li>);
      return;
    }
    flush();
    blocks.push(<p key={`p-${i}`}>{inline(line, sources, onCite, `p${i}`)}</p>);
  });
  flush();
  return <>{blocks}</>;
}
