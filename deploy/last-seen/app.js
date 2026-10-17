// All URLs are RELATIVE (no leading slash): "/x" would hit the VSS UI, not this app.
const $ = (id) => document.getElementById(id);
let appToken = '';

async function api(method, path, body) {
  const opt = { method, headers: {} };
  if (appToken) opt.headers['X-App-Token'] = appToken;
  if (body !== undefined && method !== 'GET') {
    opt.headers['Content-Type'] = 'application/json';
    opt.body = typeof body === 'string' ? body : JSON.stringify(body);
  }
  const r = await fetch(path.replace(/^\//, ''), opt);
  const text = await r.text();
  let data; try { data = JSON.parse(text); } catch { data = text; }
  return { status: r.status, data };
}

// Long calls: POST jobs/<api path> -> {job_id}; poll jobs/<id> (dodges the nginx 60 s upstream timeout).
async function apiJob(method, path, body, { every = 2000, timeoutMs = 600000 } = {}) {
  const start = await api(method, 'jobs/' + path.replace(/^\//, ''), body);
  if (start.status !== 202) return start;
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    await new Promise((res) => setTimeout(res, every));
    const j = await api('GET', 'jobs/' + start.data.job_id);
    if (j.data.status !== 'running') return { status: j.data.http_status, data: j.data.result };
  }
  return { status: 504, data: 'job timed out' };
}

function videoUrl(source, startSec = 0) {
  return 'video?source=' + encodeURIComponent(source) + (startSec ? '#t=' + startSec : '');
}

const state = {
  query: '',
  chunks: [],
  selected: null,
  activeSeg: null,
  busy: false,
};

const HL = [
  { cls: 'hl-people', re: /\b(people|person|persons|man|men|woman|women|boy|girl|child|children|crowd|pedestrian|pedestrians|someone|anybody|witness|witnesses)\b/gi },
  { cls: 'hl-clothing', re: /\b(shirt|shirts|pants|trousers|jeans|jacket|jackets|coat|coats|skirt|skirts|dress|dresses|hoodie|sweater|sweaters|blouse|blouses|suit|suits|scarf|hat|hats|cap|caps|shoes|sneakers|boots|socks|tie|ties|gloves|khaki|clothing|clothes|outfit|wear(?:ing)?|apparel)\b/gi },
  { cls: 'hl-phone', re: /\b(phone|phones|cellphone|cellphones|smartphone|smartphones|mobile|mobiles|iphone|android)\b/gi },
  { cls: 'hl-bag', re: /\b(bag|bags|handbag|handbags|purse|purses|backpack|backpacks|tote|totes|luggage|suitcase|briefcases?)\b/gi },
  { cls: 'hl-bike', re: /\b(bicycle|bicycles|bike|bikes|cyclist|cyclists|cycling)\b/gi },
];

function escapeHtml(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/** Collapse consecutive repeated sentences into one + "(repeated xN)". */
function collapseRepeats(text) {
  const raw = String(text || '').trim();
  if (!raw) return '';
  const parts = raw.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [raw];
  const sentences = parts.map((p) => p.trim()).filter(Boolean);
  if (!sentences.length) return raw;
  const out = [];
  let i = 0;
  while (i < sentences.length) {
    const cur = sentences[i];
    let j = i + 1;
    while (j < sentences.length && sentences[j] === cur) j++;
    const n = j - i;
    out.push(n > 1 ? `${cur} (repeated x${n})` : cur);
    i = j;
  }
  return out.join(' ');
}

function highlightText(text) {
  const collapsed = collapseRepeats(text);
  if (!collapsed) return '';
  // Tokenize into words / non-words so overlapping category matches stay stable.
  return collapsed.replace(/([A-Za-z']+)|([^A-Za-z']+)/g, (all, word, other) => {
    if (other) return escapeHtml(other);
    for (const { cls, re } of HL) {
      re.lastIndex = 0;
      if (re.test(word)) {
        re.lastIndex = 0;
        return `<mark class="hl ${cls}">${escapeHtml(word)}</mark>`;
      }
    }
    return escapeHtml(word);
  });
}

function fmtSec(sec) {
  const n = Number(sec);
  if (!Number.isFinite(n)) return '—';
  const s = Math.max(0, Math.floor(n));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${String(r).padStart(2, '0')}`;
}

function fmtTime(iso) {
  if (!iso) return '—';
  try {
    const d = new Date(iso.endsWith('Z') || /[+-]\d{2}:\d{2}$/.test(iso) ? iso : iso + 'Z');
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
  } catch {
    return String(iso);
  }
}

function peopleCount(objectCounts) {
  let counts = objectCounts;
  if (counts == null || counts === '') return null;
  if (typeof counts === 'string') {
    try { counts = JSON.parse(counts); } catch { return null; }
  }
  if (typeof counts !== 'object' || Array.isArray(counts)) return null;
  const keys = ['person', 'people', 'man', 'woman', 'child', 'pedestrian'];
  let total = 0;
  let found = false;
  for (const [k, v] of Object.entries(counts)) {
    if (keys.includes(String(k).toLowerCase())) {
      const n = Number(v);
      if (Number.isFinite(n)) { total += n; found = true; }
    }
  }
  return found ? total : null;
}

function simPct(score) {
  const n = Number(score);
  if (!Number.isFinite(n)) return '—';
  return `${Math.round(n * 100)}%`;
}

function setBusy(busy) {
  state.busy = busy;
  $('find').disabled = busy;
  $('report-btn').disabled = busy || !state.selected;
  document.querySelectorAll('button.example').forEach((b) => { b.disabled = busy; });
}

function setStatus(msg, working = false) {
  $('status').textContent = msg || '';
  $('status').classList.toggle('working', working);
}

async function health() {
  try {
    const r = await api('GET', 'health');
    $('health').textContent = r.status === 200 ? 'healthy' : 'unhealthy ' + r.status;
    $('health').className = 'pill ' + (r.status === 200 ? 'ok' : 'bad');
  } catch {
    $('health').textContent = 'offline';
    $('health').className = 'pill bad';
  }
}

function renderSynth(llm) {
  const el = $('synth');
  if (!llm) { el.hidden = true; el.innerHTML = ''; return; }
  el.hidden = false;
  if (llm.error) {
    el.className = 'synth error';
    el.innerHTML = `<h2>Synthesis</h2><div>${escapeHtml(llm.error)}</div>`;
    return;
  }
  el.className = 'synth';
  el.innerHTML = `<h2>What we found</h2><div>${highlightText(llm.response || '')}</div>`;
}

function renderCards() {
  const root = $('cards');
  root.innerHTML = '';
  const chunks = state.chunks || [];
  $('empty').hidden = chunks.length > 0 || state.busy;
  if (!chunks.length) return;

  for (const chunk of chunks) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'card' + (state.selected === chunk ? ' selected' : '');
    const start = chunk.best_match_start_sec;
    const end = chunk.best_match_end_sec;
    btn.innerHTML = `
      <video muted playsinline preload="metadata" src="${escapeHtml(videoUrl(chunk.preview_source))}"></video>
      <div class="card-body">
        <p class="fname">${escapeHtml(chunk.filename || 'clip')}</p>
        <div class="meta">
          <span>${escapeHtml(chunk.camera_id || 'camera ?')}</span>
          <span>${escapeHtml(fmtTime(chunk.upload_timestamp))}</span>
          <span>${escapeHtml(fmtSec(start))}–${escapeHtml(fmtSec(end))}</span>
          <span class="sim">${escapeHtml(simPct(chunk.similarity_score))}</span>
        </div>
        <div class="reason">${highlightText(chunk.reasoning_content || '')}</div>
      </div>`;
    btn.addEventListener('click', () => openChunk(chunk));
    root.appendChild(btn);
  }
}

async function ensureReasoning(seg) {
  if (seg.reasoning_content) return seg.reasoning_content;
  if (!seg.source) return '';
  if (seg._fetching) return seg._fetching;
  seg._fetching = (async () => {
    try {
      const r = await api('GET', 'api/v1/videos/metadata?source=' + encodeURIComponent(seg.source));
      if (r.status === 200 && r.data) {
        seg.reasoning_content = r.data.reasoning_content || '';
        if (r.data.object_counts != null) seg.object_counts = r.data.object_counts;
      }
    } catch { /* leave empty */ }
    return seg.reasoning_content || '';
  })();
  return seg._fetching;
}

function playSource(source, startSec = 0) {
  const player = $('player');
  if (!source) return;
  player.src = videoUrl(source, startSec);
  player.play().catch(() => {});
}

async function openChunk(chunk) {
  state.selected = chunk;
  state.activeSeg = null;
  $('detail').hidden = false;
  $('report').hidden = true;
  $('detail-title').textContent = chunk.filename || 'Moment';
  $('report-btn').disabled = state.busy;
  renderCards();

  const timeline = Array.isArray(chunk.timeline) ? chunk.timeline : [];
  const best = timeline.find((s) => s.is_best_match)
    || timeline.find((s) => s.segment_number === chunk.best_segment_number)
    || null;

  playSource(chunk.preview_source || best?.source, chunk.best_match_start_sec || 0);

  const bar = $('timeline');
  bar.innerHTML = '';
  for (const seg of timeline) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'tl-seg' + (seg.is_best_match ? ' best' : '');
    b.title = `${fmtSec(seg.segment_start_sec)}–${fmtSec(seg.segment_end_sec)}`;
    b.innerHTML = `<span class="tip">${escapeHtml(fmtSec(seg.segment_start_sec))}–${escapeHtml(fmtSec(seg.segment_end_sec))}</span>`;
    b.addEventListener('click', () => selectSegment(seg));
    bar.appendChild(b);
  }

  const list = $('seg-list');
  list.innerHTML = '';
  for (const seg of timeline) {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'seg-item' + (seg.is_best_match ? ' best' : '');
    item.dataset.seg = String(seg.segment_number);
    const people = peopleCount(seg.object_counts);
    item.innerHTML = `
      <div class="seg-top">
        <span class="time">${escapeHtml(fmtSec(seg.segment_start_sec))}–${escapeHtml(fmtSec(seg.segment_end_sec))}</span>
        <span class="people">${people == null ? 'people: —' : 'people: ' + people}</span>
        ${seg.is_best_match ? '<span class="sim">best match</span>' : ''}
      </div>
      <div class="seg-body">${seg.reasoning_content ? highlightText(seg.reasoning_content) : '<span style="color:var(--muted)">Loading description…</span>'}</div>`;
    item.addEventListener('click', () => selectSegment(seg));
    list.appendChild(item);

    if (!seg.reasoning_content) {
      ensureReasoning(seg).then(() => {
        const body = item.querySelector('.seg-body');
        const p = peopleCount(seg.object_counts);
        const peopleEl = item.querySelector('.people');
        if (peopleEl) peopleEl.textContent = p == null ? 'people: —' : 'people: ' + p;
        if (body) body.innerHTML = highlightText(seg.reasoning_content || '(no description)');
      });
    }
  }

  if (best) selectSegment(best, false);
}

function selectSegment(seg, play = true) {
  state.activeSeg = seg;
  document.querySelectorAll('.tl-seg').forEach((el, i) => {
    const timeline = state.selected?.timeline || [];
    el.classList.toggle('active', timeline[i] === seg);
  });
  document.querySelectorAll('.seg-item').forEach((el) => {
    el.classList.toggle('active', el.dataset.seg === String(seg.segment_number));
  });
  if (play && seg.source) playSource(seg.source, 0);
}

async function doSearch() {
  const query = $('query').value.trim();
  if (!query) { setStatus('Enter a description first.'); return; }
  state.query = query;
  setBusy(true);
  setStatus('Searching…', true);
  $('detail').hidden = true;
  $('report').hidden = true;
  state.selected = null;
  state.chunks = [];
  renderCards();
  renderSynth(null);

  try {
    const r = await apiJob('POST', 'api/v1/search', {
      query,
      top_k: 12,
      llm_top_n: 3,
      min_similarity: 0.1,
    });
    if (r.status !== 200) {
      const detail = typeof r.data === 'object' && r.data?.detail ? r.data.detail : r.data;
      setStatus(`Search failed (${r.status}): ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`);
      return;
    }
    state.chunks = r.data.chunk_results || [];
    renderSynth(r.data.llm_synthesis);
    renderCards();
    setStatus(state.chunks.length
      ? `${state.chunks.length} moment${state.chunks.length === 1 ? '' : 's'} found`
      : 'No matching moments. Try different wording or check whether ingestion is still running.');
  } catch (e) {
    setStatus('Search error: ' + (e.message || e));
  } finally {
    setBusy(false);
  }
}

async function doReport() {
  if (!state.selected) return;
  const searchText = state.query || $('query').value.trim();
  const question =
    `Someone reported: "${searchText}". Give a chronological timeline with timestamps and list every other person nearby with clothing, items and actions. Call them potential witnesses; do not accuse anyone.`;
  setBusy(true);
  setStatus('Generating report…', true);
  $('report').hidden = false;
  $('report-body').textContent = 'Working…';

  try {
    const r = await apiJob('POST', 'api/v1/agent/ask', {
      question,
      original_video: state.selected.original_video,
      top_k: 10,
    });
    if (r.status !== 200) {
      const detail = typeof r.data === 'object' && r.data?.detail ? r.data.detail : r.data;
      $('report-body').textContent = `Report failed (${r.status}): ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`;
      setStatus('Report failed.');
      return;
    }
    const answer = r.data?.answer ?? (typeof r.data === 'string' ? r.data : JSON.stringify(r.data));
    $('report-body').innerHTML = highlightText(answer);
    setStatus('Report ready.');
  } catch (e) {
    $('report-body').textContent = 'Report error: ' + (e.message || e);
    setStatus('Report error.');
  } finally {
    setBusy(false);
  }
}

$('find').onclick = doSearch;
$('report-btn').onclick = doReport;
$('query').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) doSearch();
});
document.querySelectorAll('button.example').forEach((btn) => {
  btn.addEventListener('click', () => {
    $('query').value = btn.dataset.q || btn.textContent;
    doSearch();
  });
});

health();
setInterval(health, 15000);
$('empty').hidden = false;
