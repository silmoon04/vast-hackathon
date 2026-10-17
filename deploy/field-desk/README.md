# Video Atlas Field Desk

An on-cluster application that filters the team's real VSS catalogue, plays a
selected chunk, and computes Pinch Point occupancy analysis from its detection
sidecars. The question interface calls these analysis tools directly.

The interface supports people or vehicle counts, busiest and quietest seconds,
literal sustained-count thresholds, time ranges, coverage checks, selectable
image regions, detection overlays, timestamp evidence and JSON export. It does
not count unique identities or answer arbitrary questions about a scene.

Open **Project guide** for simple explanations of all 16 apps and prototypes,
practical examples, and the VAST inputs and custom processing each one needs.
The guide links to animated demos and implementation notes. Pinch Point's
occupancy method is connected in this Field Desk; the other projects retain
their local-demo status and describe proposed pipeline connections. Street
Time Machine's ten tools appear within its single project entry.

The calculation follows `prototypes/13_pinch_point/run.py`: count box
bottom-centres inside a polygon, take the rounded median for each second, and
require every second of a sustained interval to meet the threshold. Missing
sidecars remain unknown. The platform's YOLO11 model supplies detections; the
prototype's local YOLO26 model is not loaded.

## Deploy

Run from the lab VM after loading the single team config in `/config/`:

```sh
NS="$USERNAME" \
APP_HOST="video-lab-team-${USERNAME#team-}.cosmos.vastdata.com" \
APP_NAME=atlas-field-desk \
VSS_URL="http://video-backend-service:8000" \
bash deploy/deploy.sh deploy/field-desk
```

The app uses Python's standard library, so no package installation or GPU is
required. `VSS_URL`, `VSS_USERNAME` and `VSS_PASSWORD` come from a Kubernetes
Secret through the deployment script. Browser requests never receive the VSS
password or JWT. Video playback is restricted to sources discovered through the
authenticated catalogue; Range requests are proxied server-side.

Use the internal backend service above for this namespace. The ingress hostname
resolves on the lab VM but did not resolve inside the deployed application pod.

The deployment script exposes the application at `/app/`. When another app
uses that route, Field Desk can have its own `/field-desk/` Ingress pointing
to the same service. Asset and API routes are relative, so both prefixes work.
The application exposes `/health` for readiness. The project guide opens at
`#projects` after either prefix.

The catalogue is cached for five minutes. Up to 16 analyses and 64 segment lists
are retained in pod memory. A pod replacement clears these caches, and opening
a clip recomputes them. No footage, model weights or derived records are written
to the shared ingestion pipeline or team database.

## Checks

```sh
python -m pytest tests/test_field_desk.py -q
```

The tests cover median rounding, empty versus missing detections, segment
offsets, duplicate frames, polygon geometry, confidence filtering, sustained
thresholds, partial seconds and supported question intents. Validate real
catalogue filtering, sidecar retrieval, video seeking and questions after a
deployment. Local experiment accuracy figures are not a validation of YOLO11.
