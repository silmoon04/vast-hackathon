"""Serve Time Search from a prepared source bundle and licensed scene assets."""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import tarfile
from pathlib import Path

for key in ['OPENCV_FFMPEG_THREADS', 'OPENCV_FOR_THREADS_NUM', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ.setdefault(key, '2')

DATA = Path(os.environ.get('STM_DATA', '/data'))
RUNTIME = Path('/tmp/stm-runtime')
RUNTIME.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(Path(__file__).with_name('source.tgz.b64').read_bytes())), mode='r:gz') as archive:
    archive.extractall(RUNTIME, filter='data')

os.environ.setdefault('STM_OUT', str(DATA / 'out'))
os.environ.setdefault('STM_TRACK_CACHE', str(DATA / 'tracks'))
os.environ.setdefault('STM_CAMERAS', str(DATA / 'cams.json'))
os.environ.setdefault('STM_WARM', '0')
os.environ.setdefault('STM_FIND_MODELS', 'deepseek-ai/DeepSeek-V4.1-Flash')
os.environ.setdefault('STM_FIND_TIMEOUT', '12')
private_config = DATA / 'private' / 'model.json'
if private_config.is_file():
    for key, value in json.loads(private_config.read_text()).items():
        if key in {'WANDB_API_KEY', 'WANDB_TEAM', 'WANDB_PROJECT'}:
            os.environ.setdefault(key, value)

scene_root = DATA / 'time-search' / 'scenes'
scene_root.mkdir(parents=True, exist_ok=True)
(DATA / 'out').mkdir(parents=True, exist_ok=True)
scene_link = RUNTIME / 'ui' / 'demo' / 'scenes'
if scene_root.is_dir() and not scene_link.exists():
    scene_link.symlink_to(scene_root, target_is_directory=True)

sys.path.insert(0, str(RUNTIME))
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, Response  # noqa: E402
from stm import agent, find, server  # noqa: E402

# The provider key stays on the server. The browser receives only a parsed tool
# call; timestamps, candidates and evidence come from the prepared scene index.
def provider_post(body: dict, timeout: float) -> dict:
    config = agent.load_config()
    if config is None:
        raise RuntimeError('Model credentials are unavailable')
    headers = {'Authorization': f'Bearer {config.key}', 'OpenAI-Project': config.project_id}
    return agent.post_json(agent.API_URL, body, headers, timeout)

find._post = provider_post
app = FastAPI(title='Time Search', lifespan=server.lifespan)
PREFIX = os.environ.get('STM_PUBLIC_PREFIX', '/app').rstrip('/')

def public_text(text: str) -> str:
    for quote in ('"', "'", '`'):
        for path in ('/api/', '/out/'):
            text = text.replace(quote + path, quote + PREFIX + path)
    return text

@app.get('/health')
def health():
    scenes = [name for name in ('meva', 'caviar', 'lab') if (scene_root / name / 'scene.json').is_file()]
    if len(scenes) != 3:
        raise HTTPException(503, 'Prepared scene assets are incomplete')
    return {'ok': True, 'app': 'Time Search', 'scenes': scenes, 'model': find.MODELS[0],
            'model_configured': agent.load_config() is not None}

@app.get('/api/find/health')
def find_health():
    return {'ok': agent.load_config() is not None, 'relay': False,
            'provider': 'W&B Inference', 'models': find.MODELS}

@app.get('/', response_class=HTMLResponse)
@app.get('/demo.html', response_class=HTMLResponse)
def index():
    text = (RUNTIME / 'ui' / 'demo.html').read_text(encoding='utf-8')
    text = text.replace('<head>', '<head><base href="' + PREFIX + '/">', 1)
    return public_text(text)

@app.get('/demo/{path:path}')
def demo_asset(path: str):
    base = (RUNTIME / 'ui' / 'demo').resolve()
    target = (base / path).resolve()
    roots = (base, scene_root.resolve())
    if not any(target == root or root in target.parents for root in roots) or not target.is_file():
        raise HTTPException(404, 'Asset not found')
    if target.suffix in {'.js', '.mjs', '.css'}:
        media = 'text/css' if target.suffix == '.css' else 'application/javascript'
        return Response(public_text(target.read_text(encoding='utf-8')), media_type=media)
    return FileResponse(target)

app.mount('/', server.app)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', '8080')))
