"""Update Time Search after backing up the current pod's ephemeral data.

Run on the cluster-accessible host, beside main.py, requirements.txt,
source.tgz.b64 and scenes.tgz. Credentials are read privately from environment
variables or /etc/environment; no credential values are logged.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--namespace', required=True)
parser.add_argument('--kubeconfig', required=True)
args = parser.parse_args()
env = dict(os.environ, KUBECONFIG=args.kubeconfig)
k = ['kubectl', '-n', args.namespace]
base = Path(__file__).resolve().parent

def run(command, payload=None, check=True):
    return subprocess.run(k + command, input=payload, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env=env, check=check)

files = ['main.py', 'requirements.txt', 'source.tgz.b64']
document = {'apiVersion': 'v1', 'kind': 'ConfigMap',
            'metadata': {'name': 'street-time-machine-code'},
            'data': {name: (base / name).read_text() for name in files}}
if sum(len(value.encode()) for value in document['data'].values()) > 900_000:
    raise SystemExit('Source bundle exceeds the ConfigMap budget')
run(['apply', '--server-side', '--force-conflicts', '--field-manager=street-time-machine', '-f', '-'], json.dumps(document).encode())
run(['exec', 'deploy/street-time-machine', '--', 'mkdir', '-p', '/data/time-search', '/data/private'])
run(['exec', '-i', 'deploy/street-time-machine', '--', 'tar', 'xzf', '-', '-C', '/data/time-search'],
    (base / 'scenes.tgz').read_bytes())

keys = {'WANDB_API_KEY', 'WANDB_TEAM', 'WANDB_PROJECT'}
credentials = {key: os.environ[key] for key in keys if os.environ.get(key)}
if Path('/etc/environment').is_file():
    for line in Path('/etc/environment').read_text().splitlines():
        try:
            parts = shlex.split(line)
        except ValueError:
            continue
        for part in parts:
            key, sep, value = part.partition('=')
            if sep and key in keys and value:
                credentials.setdefault(key, value)
if set(credentials) != keys:
    raise SystemExit('Server-side W&B credentials are incomplete; no restart performed')
writer = "import os,pathlib,sys; p=pathlib.Path('/data/private/model.json'); p.write_bytes(sys.stdin.buffer.read()); os.chmod(p,0o600)"
run(['exec', '-i', 'deploy/street-time-machine', '--', 'python', '-c', writer], json.dumps(credentials).encode())

# The backup includes existing prepared outputs and video, so a rollout does not
# discard earlier work. Keep this file private: it also contains server config.
backup = base / ('data-backup-' + time.strftime('%Y%m%d-%H%M%S') + '.tar')
with backup.open('wb') as stream:
    os.chmod(backup, 0o600)
    subprocess.run(k + ['exec', 'deploy/street-time-machine', '--', 'tar', 'cf', '-', '-C', '/data', '.'],
                   stdout=stream, stderr=subprocess.PIPE, env=env, check=True)
old_pods = json.loads(run(['get', 'pods', '-l', 'app=street-time-machine', '-o', 'json']).stdout)
old_names = {pod['metadata']['name'] for pod in old_pods['items']}
run(['rollout', 'restart', 'deployment/street-time-machine'])
for attempt in range(60):
    pods = json.loads(run(['get', 'pods', '-l', 'app=street-time-machine', '-o', 'json']).stdout)
    new_names = [pod['metadata']['name'] for pod in pods['items']
                 if pod['metadata']['name'] not in old_names and not pod['metadata'].get('deletionTimestamp')]
    if not new_names:
        time.sleep(3)
        continue
    target = new_names[0]
    probe = run(['exec', target, '--', 'python', '-c', 'print(1)'], check=False)
    if probe.returncode == 0:
        # Restore the complete snapshot, including model credentials and scenes.
        with backup.open('rb') as stream:
            restored = subprocess.run(k + ['exec', '-i', target, '--', 'tar', 'xf', '-', '-C', '/data'],
                                      stdin=stream, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        if restored.returncode == 0:
            break
    time.sleep(3)
else:
    raise SystemExit('New pod did not accept the prepared data; backup retained on disk')
print('Updated Time Search; previous app data is backed up and restored.')
