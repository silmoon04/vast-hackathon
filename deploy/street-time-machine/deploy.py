"""Deploy STM code and its persistent data volume; publish only after asset checks.

Run on a cluster-accessible host:
python3 deploy.py --phase prepare --namespace <namespace>
Copy prepared assets into /data, writing ready.json last, then:
python3 deploy.py --phase publish --namespace <namespace>
"""
import argparse
import base64
import json
import os
import re
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--phase', choices=['prepare', 'publish'], required=True)
parser.add_argument('--namespace', required=True,
                    help='Kubernetes namespace for the application')
parser.add_argument('--kubeconfig', type=Path,
                    help='kubeconfig path (default: /config/<namespace>-k8s.yaml)')
parser.add_argument('--ingress-name', default='street-time-machine')
parser.add_argument('--host', help='ingress hostname (default follows the VAST challenge cluster convention)')
parser.add_argument('--storage', choices=['ephemeral', 'persistent'], default='ephemeral')
args = parser.parse_args()
if not re.fullmatch(r'[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?', args.namespace) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,62}', args.ingress_name):
    parser.error('Use a DNS-compatible namespace and ingress name')
name = 'street-time-machine'
base = Path(__file__).resolve().parent
env = dict(os.environ, KUBECONFIG=str(args.kubeconfig or Path('/config') / f'{args.namespace}-k8s.yaml'))
k = ['kubectl', '-n', args.namespace]


def run(command, document=None):
    return subprocess.run(k + command, input=None if document is None else json.dumps(document),
                          text=True, env=env, check=True)


def apply(document):
    run(['apply', '--server-side', '--field-manager=street-time-machine', '-f', '-'], document)


if args.phase == 'prepare':
    files = ['main.py', 'requirements.txt', 'source.tgz.b64']
    contents = {file: (base / file).read_text() for file in files}
    if sum(len(value.encode()) for value in contents.values()) > 900_000:
        raise SystemExit('App bundle exceeds the ConfigMap size budget')
    apply({'apiVersion': 'v1', 'kind': 'ConfigMap', 'metadata': {'name': name + '-code'}, 'data': contents})
    if args.storage == 'persistent':
        apply({'apiVersion': 'v1', 'kind': 'PersistentVolumeClaim', 'metadata': {'name': name + '-data'},
               'spec': {'accessModes': ['ReadWriteOnce'], 'resources': {'requests': {'storage': '2Gi'}}}})
    storage = {'persistentVolumeClaim': {'claimName': name + '-data'}} if args.storage == 'persistent' else {'emptyDir': {'sizeLimit': '2Gi'}}
    apply({'apiVersion': 'apps/v1', 'kind': 'Deployment', 'metadata': {'name': name, 'labels': {'app': name}},
           'spec': {'replicas': 1, 'selector': {'matchLabels': {'app': name}}, 'strategy': {'type': 'Recreate'},
                    'template': {'metadata': {'labels': {'app': name}}, 'spec': {'containers': [{
                        'name': 'app', 'image': 'python:3.12-slim', 'imagePullPolicy': 'IfNotPresent',
                        'workingDir': '/code', 'command': ['bash', '-c'],
                        'args': ['set -euo pipefail; pip install --no-cache-dir -q -r requirements.txt; exec python -B main.py'],
                        'env': [{'name': 'PORT', 'value': '8080'}, {'name': 'PYTHONUNBUFFERED', 'value': '1'},
                                {'name': 'STM_STORAGE', 'value': args.storage}],
                        'ports': [{'containerPort': 8080}],
                        'resources': {'requests': {'cpu': '250m', 'memory': '512Mi'},
                                      'limits': {'cpu': '2', 'memory': '2Gi'}},
                        'readinessProbe': {'httpGet': {'path': '/health', 'port': 8080},
                                           'initialDelaySeconds': 10, 'periodSeconds': 5},
                        'volumeMounts': [{'name': 'code', 'mountPath': '/code', 'readOnly': True},
                                         {'name': 'data', 'mountPath': '/data'}]}],
                        'volumes': [{'name': 'code', 'configMap': {'name': name + '-code'}},
                                    {'name': 'data', **storage}]}}}})
    apply({'apiVersion': 'v1', 'kind': 'Service', 'metadata': {'name': name, 'labels': {'app': name}},
           'spec': {'selector': {'app': name}, 'ports': [{'port': 80, 'targetPort': 8080}]}})
    run(['rollout', 'restart', 'deployment/' + name])
else:
    run(['rollout', 'status', 'deployment/' + name, '--timeout=180s'])
    # Reconcile the configured ingress route for this application.
    run(['apply', '-f', '-'], {'apiVersion': 'networking.k8s.io/v1', 'kind': 'Ingress',
           'metadata': {'name': args.ingress_name, 'labels': {'app': name},
                        'annotations': {'nginx.ingress.kubernetes.io/rewrite-target': '/$2'}},
           'spec': {'ingressClassName': 'nginx', 'rules': [{
               'host': args.host or 'video-lab-' + args.namespace + '.cosmos.vastdata.com',
               'http': {'paths': [{'path': '/app(/|$)(.*)', 'pathType': 'ImplementationSpecific',
                                  'backend': {'service': {'name': name, 'port': {'number': 80}}}}]}}]}})
