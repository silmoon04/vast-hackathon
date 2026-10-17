"""Deploy Find Desk on its own route using the team's existing VSS Secret."""
import argparse
import json
from pathlib import Path
import subprocess


def resources(root, dataset, namespace, host, secret, speech_secret, agent_secret):
    name = "atlas-find-desk"
    labels = {"app": name}
    data = {file: (root / file).read_text(encoding="utf-8-sig")
            for file in ("main.py", "index.html", "app.js", "style.css")}
    if (root / "planner.py").is_file():
        data["planner.py"] = (root / "planner.py").read_text(encoding="utf-8-sig")
    data["dataset.json"] = json.dumps(dataset)
    if sum(len(value.encode()) for value in data.values()) > 850000:
        raise ValueError("The application bundle is too large for this deployment.")
    metadata = {"name": name, "namespace": namespace, "labels": labels}
    return {"apiVersion": "v1", "kind": "List", "items": [
        {"apiVersion": "v1", "kind": "ConfigMap", "metadata": metadata, "data": data},
        {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": metadata,
         "spec": {"replicas": 1, "selector": {"matchLabels": labels}, "template": {
             "metadata": {"labels": labels}, "spec": {"containers": [{"name": "app", "image": "python:3.12-slim",
             "command": ["python", "/app/main.py"], "ports": [{"containerPort": 8080}],
             "env": [{"name": "PORT", "value": "8080"}, {"name": "PYTHONUNBUFFERED", "value": "1"}] +
                    [{"name": key, "valueFrom": {"secretKeyRef": {"name": secret, "key": key}}}
                     for key in ("VSS_URL", "VSS_USERNAME", "VSS_PASSWORD")] +
                    [{"name": key, "valueFrom": {"secretKeyRef": {"name": speech_secret, "key": key, "optional": True}}}
                     for key in ("CANARY_URL", "GPU_BEARER_TOKEN")] +
                    [{"name": key, "valueFrom": {"secretKeyRef": {"name": agent_secret, "key": key, "optional": True}}}
                     for key in ("WANDB_API_KEY", "WANDB_PROJECT_PATH")],
             "resources": {"requests": {"cpu": "100m", "memory": "96Mi"}, "limits": {"cpu": "500m", "memory": "256Mi"}},
             "readinessProbe": {"httpGet": {"path": "/health", "port": 8080}, "initialDelaySeconds": 2, "periodSeconds": 5},
             "volumeMounts": [{"name": "code", "mountPath": "/app", "readOnly": True}]}],
             "volumes": [{"name": "code", "configMap": {"name": name}}]}}}},
        {"apiVersion": "v1", "kind": "Service", "metadata": metadata,
         "spec": {"selector": labels, "ports": [{"port": 80, "targetPort": 8080}]}},
        {"apiVersion": "networking.k8s.io/v1", "kind": "Ingress", "metadata": {**metadata,
         "annotations": {"nginx.ingress.kubernetes.io/use-regex": "true", "nginx.ingress.kubernetes.io/rewrite-target": "/$2",
                         "nginx.ingress.kubernetes.io/proxy-read-timeout": "360"}},
         "spec": {"ingressClassName": "nginx", "rules": [{"host": host, "http": {"paths": [
             {"path": "/find(/|$)(.*)", "pathType": "ImplementationSpecific",
              "backend": {"service": {"name": name, "port": {"number": 80}}}}]}}]}}
    ]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--namespace", default="team-13")
    parser.add_argument("--secret", default="atlas-field-desk-vss-creds")
    parser.add_argument("--speech-secret", default="atlas-find-desk-speech")
    parser.add_argument("--agent-secret", default="atlas-find-desk-agent")
    parser.add_argument("--kubeconfig", default="/config/team-13-k8s.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    bundle = resources(root, dataset, args.namespace, args.host, args.secret, args.speech_secret, args.agent_secret)
    base = ["kubectl", "--kubeconfig=" + args.kubeconfig, "-n", args.namespace]
    subprocess.run(base + ["apply", "-f", "-"], input=json.dumps(bundle), text=True, check=True)
    subprocess.run(base + ["rollout", "restart", "deployment/atlas-find-desk"], check=True)
    subprocess.run(base + ["rollout", "status", "deployment/atlas-find-desk", "--timeout=120s"], check=True)
