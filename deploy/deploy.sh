#!/usr/bin/env bash
# Deploy deploy/app (or any FLAT app dir) to the team namespace at /app, following the
# deploy-app-no-registry contract: ConfigMap code + Secret creds + python:3.12-slim Deployment,
# Service :80 -> 8080, nginx Ingress path /app(/|$)(.*) on the team host. Run on the lab VM.
#
#   NS=<team> APP_HOST=video-lab-team-<N>.cosmos.vastdata.com bash deploy/deploy.sh [APP_DIR]
#
# Call it through bash: a commit made on Windows records this file without the exec bit, so
# ./deploy/deploy.sh gives "Permission denied" on a fresh Linux clone.
#
# Reads VSS_URL / VSS_USERNAME / VSS_PASSWORD (or INGRESS_URL / USERNAME / PASSWORD from the
# sourced /config/*.config) from the environment; never prints them.
# Optional: APP_NAME (default team-app), ALLOW_WRITES=1, INGRESS_TUNING=1 (adds body-size and
# read-timeout annotations; only if the organisers allow annotations), EXTRA_SECRET_KEYS="WANDB_API_KEY ...".
# EXTRA_SECRET_KEYS names variables set in this shell; each goes into the Secret and the pod env.
# deploy/app needs APP_TOKEN in the pod for writes (ALLOW_WRITES=1 alone keeps them off) and streams /video
# only from VIDEO_BUCKETS (unset = /video is off). VIDEO_BUCKETS defaults to $S3_SEGMENTS_BUCKET from the
# sourced config and, when set, is added to EXTRA_SECRET_KEYS, e.g.
#   APP_TOKEN=<random> ALLOW_WRITES=1 EXTRA_SECRET_KEYS="APP_TOKEN" NS=... APP_HOST=... bash deploy/deploy.sh
set -euo pipefail

APP_DIR="${1:-$(dirname "$0")/app}"
APP_NAME="${APP_NAME:-team-app}"
APP_PORT=8080
NS="${NS:-${USERNAME:-}}"
: "${NS:?set NS (team namespace)}"
: "${APP_HOST:?set APP_HOST (video-lab-team-<N>.cosmos.vastdata.com)}"
VSS_URL="${VSS_URL:-${INGRESS_URL:-}}"
VSS_USERNAME="${VSS_USERNAME:-${USERNAME:-}}"
VSS_PASSWORD="${VSS_PASSWORD:-${PASSWORD:-}}"
: "${VSS_URL:?VSS_URL/INGRESS_URL unset}" "${VSS_USERNAME:?VSS_USERNAME unset}" "${VSS_PASSWORD:?VSS_PASSWORD unset}"
VIDEO_BUCKETS="${VIDEO_BUCKETS:-${S3_SEGMENTS_BUCKET:-}}"
case " ${EXTRA_SECRET_KEYS:-} " in
  *" VIDEO_BUCKETS "*) ;;
  *) [ -z "$VIDEO_BUCKETS" ] || EXTRA_SECRET_KEYS="${EXTRA_SECRET_KEYS:-} VIDEO_BUCKETS" ;;
esac
export KUBECONFIG="${KUBECONFIG:-$(ls /config/*k8s*.yaml 2>/dev/null | head -1)}"
K="kubectl -n $NS"

# --- flat-dir + size checks (ConfigMap --from-file=<dir> ignores subdirectories; ~1 MiB cap) ---
if find "$APP_DIR" -mindepth 2 -type f -not -path '*/__pycache__/*' | grep -q .; then  # a local import leaves __pycache__
  echo "WARN: subdirectories in $APP_DIR are NOT included in the ConfigMap" >&2
fi
bytes=$(( $(find "$APP_DIR" -maxdepth 1 -type f -exec cat {} + | wc -c) ))  # portable: no GNU find -printf
if [ "$bytes" -gt 900000 ]; then echo "ERROR: app dir is $bytes bytes (> ~900 kB; ConfigMap limit is 1 MiB)" >&2; exit 1; fi
echo "app dir: $(find "$APP_DIR" -maxdepth 1 -type f | wc -l) files, $bytes bytes"

# --- code + secret ---
# Server-side apply: client-side apply copies the whole ConfigMap into the last-applied-configuration
# annotation, which is capped at 256 KiB and would fail long before the 900 kB guard above.
$K create configmap "${APP_NAME}-code" --from-file="$APP_DIR" --dry-run=client -o yaml \
  | $K apply --server-side --force-conflicts -f -
# Values go to kubectl as files in a private (0700) temp dir: --from-literal would put them on a command
# line, readable by any user through ps or /proc. printf is a builtin, so it never execs with a value.
secret_dir=$(mktemp -d)
trap 'rm -rf "$secret_dir"' EXIT
secret_args=()
put_secret() { printf '%s' "$2" > "$secret_dir/$1"; secret_args+=(--from-file="$1=$secret_dir/$1"); }
put_secret VSS_URL "$VSS_URL"
put_secret VSS_USERNAME "$VSS_USERNAME"
put_secret VSS_PASSWORD "$VSS_PASSWORD"
extra_env=""
for key in ${EXTRA_SECRET_KEYS:-}; do
  [ -n "${!key:-}" ] || { echo "WARN: $key is empty; skipped" >&2; continue; }
  put_secret "$key" "${!key}"
  extra_env+="
            - name: $key
              valueFrom: {secretKeyRef: {name: ${APP_NAME}-vss-creds, key: $key}}"
done
$K create secret generic "${APP_NAME}-vss-creds" "${secret_args[@]}" --dry-run=client -o yaml | $K apply -f - >/dev/null
echo "secret ${APP_NAME}-vss-creds applied (values not shown)"

annotations="nginx.ingress.kubernetes.io/rewrite-target: /\$2"
if [ "${INGRESS_TUNING:-0}" = "1" ]; then
  annotations+="
    nginx.ingress.kubernetes.io/proxy-body-size: 110m
    nginx.ingress.kubernetes.io/proxy-read-timeout: \"300\"
    nginx.ingress.kubernetes.io/proxy-send-timeout: \"300\""
fi

cat <<YAML | $K apply -f -
apiVersion: apps/v1
kind: Deployment
metadata: {name: ${APP_NAME}, labels: {app: ${APP_NAME}}}
spec:
  replicas: 1
  selector: {matchLabels: {app: ${APP_NAME}}}
  template:
    metadata:
      labels: {app: ${APP_NAME}}
    spec:
      containers:
        - name: app
          image: python:3.12-slim
          imagePullPolicy: IfNotPresent
          workingDir: /code
          command: ["bash", "-c"]
          args: ["set -euo pipefail; if [ -f requirements.txt ]; then pip install --no-cache-dir -q -r requirements.txt; fi; exec python main.py"]
          ports: [{containerPort: ${APP_PORT}}]
          env:
            - {name: PORT, value: "${APP_PORT}"}
            - {name: PYTHONUNBUFFERED, value: "1"}
            - {name: ALLOW_WRITES, value: "${ALLOW_WRITES:-0}"}
            - name: VSS_URL
              valueFrom: {secretKeyRef: {name: ${APP_NAME}-vss-creds, key: VSS_URL}}
            - name: VSS_USERNAME
              valueFrom: {secretKeyRef: {name: ${APP_NAME}-vss-creds, key: VSS_USERNAME}}
            - name: VSS_PASSWORD
              valueFrom: {secretKeyRef: {name: ${APP_NAME}-vss-creds, key: VSS_PASSWORD}}${extra_env}
          readinessProbe:
            httpGet: {path: /health, port: ${APP_PORT}}
            initialDelaySeconds: 10
            periodSeconds: 10
          volumeMounts: [{name: code, mountPath: /code, readOnly: true}]
      volumes:
        - name: code
          configMap: {name: ${APP_NAME}-code}
---
apiVersion: v1
kind: Service
metadata: {name: ${APP_NAME}}
spec:
  type: ClusterIP
  selector: {app: ${APP_NAME}}
  ports: [{port: 80, targetPort: ${APP_PORT}}]
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ${APP_NAME}
  annotations:
    ${annotations}
spec:
  ingressClassName: nginx
  rules:
    - host: ${APP_HOST}
      http:
        paths:
          - path: /app(/|$)(.*)
            pathType: ImplementationSpecific
            backend: {service: {name: ${APP_NAME}, port: {number: 80}}}
YAML

# Running pods never pick up ConfigMap/Secret changes: always restart.
$K rollout restart "deploy/${APP_NAME}"
$K rollout status "deploy/${APP_NAME}" --timeout=180s
echo "health: $(curl -s -o /dev/null -w '%{http_code}' "http://${APP_HOST}/app/health")  (open via the workshop portal's App button)"
