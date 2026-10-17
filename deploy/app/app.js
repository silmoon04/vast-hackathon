// All URLs are RELATIVE (no leading slash): "/x" would hit the VSS UI, not this app.
const $ = (id) => document.getElementById(id);
// Only write routes (server has ALLOW_WRITES=1 + APP_TOKEN) check X-App-Token; reads work without it.
// Kept in memory only, and the server never forwards it upstream.
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

async function health() {
  try {
    const r = await api('GET', 'health');
    $('health').textContent = r.status === 200 ? 'healthy' : 'unhealthy ' + r.status;
    $('health').className = 'pill ' + (r.status === 200 ? 'ok' : 'bad');
  } catch (e) {
    $('health').textContent = 'offline'; $('health').className = 'pill bad';
  }
}

$('apptoken').oninput = (e) => { appToken = e.target.value.trim(); };

$('send').onclick = async () => {
  $('out').textContent = '…';
  const method = $('method').value, path = $('path').value.trim();
  const raw = $('body').value.trim();
  let body;
  if (raw) { try { body = JSON.parse(raw); } catch { $('out').textContent = 'body is not valid JSON'; return; } }
  const t0 = performance.now();
  const r = $('asjob').checked ? await apiJob(method, path, body) : await api(method, path, body);
  $('out').textContent = `HTTP ${r.status} in ${((performance.now() - t0) / 1000).toFixed(2)} s\n\n` +
    (typeof r.data === 'string' ? r.data : JSON.stringify(r.data, null, 2));
};

$('play').onclick = () => {
  const s = $('source').value.trim();
  if (s) { $('player').src = videoUrl(s, parseFloat($('start').value) || 0); $('player').play().catch(() => {}); }
};

health();
setInterval(health, 15000);
