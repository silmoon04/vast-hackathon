"""Copy a prepared catalogue camera from the lab VM into the STM data volume.

Checks the current authenticated catalogue before copying any footage. Credentials
stay on the VM. The readiness marker is written only after a successful transfer.
"""
import argparse
import glob
import io
import json
import os
import shlex
import subprocess
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prepared', type=Path, default=Path.home() / 'stm')
parser.add_argument('--pack', type=Path, default=Path.home() / 'packd')
parser.add_argument('--camera', default='neighborhood_cam-1')
parser.add_argument('--config', type=Path,
                    help='platform config file (default: the only *.config file in /config)')
parser.add_argument('--kubeconfig', type=Path,
                    help='kubeconfig path (default: /config/<configured username>-k8s.yaml)')
args = parser.parse_args()
configs = [str(args.config)] if args.config else glob.glob('/config/*.config')
if len(configs) != 1:
    raise SystemExit('Expected exactly one team config')
config = {}
for line in Path(configs[0]).read_text().splitlines():
    line = line.strip().removeprefix('export ')
    if '=' in line and not line.startswith('#'):
        key, value = line.split('=', 1)
        config[key.strip()] = ' '.join(shlex.split(value, comments=True))
namespace = config['USERNAME']
if not namespace:
    raise SystemExit('The platform config must define USERNAME')
env = dict(os.environ, KUBECONFIG=str(args.kubeconfig or Path('/config') / f'{namespace}-k8s.yaml'))
k = ['kubectl', '-n', namespace]
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
base = config['INGRESS_URL'].rstrip('/')
body = json.dumps({'username': config['USERNAME'], 'password': config['PASSWORD']}).encode()
with opener.open(urllib.request.Request(base + '/api/v1/auth/login', data=body,
                 headers={'Content-Type': 'application/json'}), timeout=30) as response:
    token = json.load(response)['access_token']
clips = []
for offset in range(0, 10000, 100):
    query = urllib.parse.urlencode({'indexed': 'complete', 'limit': 100, 'offset': offset})
    with opener.open(urllib.request.Request(base + '/api/v1/videos/explore?' + query,
                     headers={'Authorization': 'Bearer ' + token}), timeout=30) as response:
        page = json.load(response)
    clips.extend(c for c in page.get('chunks', []) if c.get('camera_id') == args.camera)
    if offset + 100 >= page.get('total', 0) or not page.get('chunks'):
        break
prepared_sources = set(json.loads((args.pack / 'segments.json').read_text())['chunks'])
catalogue_sources = {c['original_video'] for c in clips}
if not prepared_sources or prepared_sources != catalogue_sources:
    raise SystemExit('Prepared sources do not exactly match the filtered current catalogue')
print(f'Catalogue verified: {len(clips)} matching chunks for {args.camera}', flush=True)

config_path = args.prepared / 'cams.local.json'
camera = json.loads(config_path.read_text())[args.camera]
video = Path(camera['video'])
if not video.is_file():
    raise SystemExit('Prepared recording is missing')
meta = args.prepared / 'out' / args.camera / 'meta.json'
if not meta.is_file() or not list((args.prepared / '_cache' / 'tracks').glob(f'{args.camera}__det__*.json.gz')):
    raise SystemExit('Prepared camera metadata or tracks are missing')
target_video = '/data/video/' + video.name
camera['video'] = target_video
manifest = {'camera': args.camera, 'catalogue_filter': {'camera_id': args.camera, 'indexed': 'complete'},
            'catalogue_matches': len(clips), 'prepared_sources_match_catalogue': True,
            'duration': json.loads(meta.read_text())['duration'], 'seeded_at': time.time(),
            'files': [c.get('filename') for c in clips],
            'analysis': 'Prepared STM modules; new questions computed from tracks in the pod',
            'detections': 'VSS YOLO11 sidecars linked by stm.tracker'}

pod_list = subprocess.check_output(k + ['get', 'pods', '-l', 'app=street-time-machine', '-o', 'json'], env=env)
pods = [p for p in json.loads(pod_list)['items'] if not p['metadata'].get('deletionTimestamp')
        and p.get('status', {}).get('phase') == 'Running']
if len(pods) != 1:
    raise SystemExit('Wait until the Street Time Machine container is Running, then seed again')
pod = pods[0]['metadata']['name']


def add_json(archive, name, content):
    raw = json.dumps(content).encode()
    item = tarfile.TarInfo(name)
    item.size, item.mode = len(raw), 0o644
    archive.addfile(item, io.BytesIO(raw))


process = subprocess.Popen(k + ['exec', '-i', pod, '--', 'tar', '-xf', '-', '-C', '/data'], env=env, stdin=subprocess.PIPE)
try:
    with tarfile.open(fileobj=process.stdin, mode='w|') as archive:
        archive.add(video, arcname='video/' + video.name, recursive=False)
        for path in sorted((args.prepared / 'out' / args.camera).rglob('*')):
            if path.is_file():
                archive.add(path, arcname=Path('out') / path.relative_to(args.prepared / 'out'), recursive=False)
        for path in (args.prepared / '_cache' / 'tracks').glob(f'{args.camera}__det__*.json.gz'):
            archive.add(path, arcname='tracks/' + path.name, recursive=False)
        add_json(archive, 'cams.json', {args.camera: camera})
    process.stdin.close()
    if process.wait() != 0:
        raise SystemExit('Asset transfer failed; readiness marker was not written')
except BaseException:
    if process.poll() is None:
        process.terminate()
    raise
# Refresh saved examples and the briefing with the deployed source. The first
# cluster check found two stale answers and a stale blind-time total.
for module in ['moments', 'ask', 'brief']:
    subprocess.run(k + ['exec', pod, '--', 'env', 'PYTHONPATH=/tmp/stm-runtime',
        'OPENCV_FFMPEG_THREADS=2', 'OPENCV_FOR_THREADS_NUM=2', 'OPENBLAS_NUM_THREADS=2', 'OMP_NUM_THREADS=2',
        'STM_OUT=/data/out', 'STM_TRACK_CACHE=/data/tracks', 'STM_CAMERAS=/data/cams.json',
        'python', '-B', '-m', 'stm.' + module, args.camera], env=env, check=True)
subprocess.run(k + ['exec', '-i', pod, '--', 'python', '-B', '-c',
    "import sys,pathlib,json;d=json.load(sys.stdin);p=pathlib.Path('/data/ready.json');p.write_text(json.dumps(d))"],
    input=json.dumps(manifest), text=True, env=env, check=True)
print('Prepared recording, evidence assets and tracks copied into the app data volume', flush=True)
