# Deploy Time Search

The hosted demo is [Time Search](https://team-13-vss.thecosmoslabs.com/app/).
It serves prepared camera footage and track evidence, then calls the supplied
DeepSeek V4.1 Flash model live to turn a request into a search or navigation tool.
The browser confirms candidates and shows the matching footage and timestamps.

The public demo offers the MEVA campus scene and NVIDIA's synthetic Lab scene.
Both retain their CC BY 4.0 credits. CAVIAR is excluded from the public route
until its redistribution terms are confirmed. Recordings, track caches and
provider credentials stay outside Git.

## Build

From the repository root:

```sh
python deploy/street-time-machine/build_bundle.py
```

The bundle includes Python, HTML, JavaScript, CSS and SVG source. Prepared scene
JSON, crops and MP4s are a separate `scenes.tgz` archive with a `scenes/` folder.
Use footage you have permission to redistribute, preserving source credits.
Copy the build output, deployment scripts and scene archive to a host with
access to your Kubernetes namespace.

## Update an existing deployment

```sh
python3 update_running.py --namespace <namespace> \
  --kubeconfig /config/<namespace>-k8s.yaml
```

This updates the app's source ConfigMap, seeds the prepared scenes, reads W&B
credentials privately from the host environment or `/etc/environment`, and
backs up the current data before replacing the serving pod. The complete data
snapshot is restored into the replacement pod. Backup archives contain private
server configuration: keep them on the deployment host and out of source control.

The server calls W&B Inference directly. Credentials never reach the browser.
`STM_FIND_MODELS` selects the provider model; its default is
`deepseek-ai/DeepSeek-V4.1-Flash`. `STM_PUBLIC_PREFIX` defaults to `/app` so
JavaScript API calls and video paths work behind the ingress rewrite.

## New deployment

```sh
python3 deploy.py --phase prepare --namespace <namespace>
python3 update_running.py --namespace <namespace> \
  --kubeconfig /config/<namespace>-k8s.yaml
python3 deploy.py --phase publish --namespace <namespace> \
  --ingress-name <your-app-ingress> --host <your-ingress-host>
```

Use the existing app ingress name when updating a published route. This avoids
creating competing `/app/` routes. The serving pod requests 250 millicores and
512 MiB, with limits of two CPUs and 2 GiB. It does not request a GPU or run
ingestion on the shared VSS graph. Prepare tracks on a separate machine.

The default volume is temporary pod storage. The update helper retains a
private backup and restores it after a rollout. For longer use, select
`--storage persistent` in the prepare step on a cluster with a working PVC.

## Verify

Check `/app/health`, both public scene manifests, MP4 range requests and
`/app/api/find/route`. The route should return the actual provider model and
tool call. The deployed UI works with typed requests. Speech and uploading a
new recording need their separate inference/indexing services configured.
