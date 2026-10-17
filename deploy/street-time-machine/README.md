# Deploy Street Time Machine to Kubernetes

This example deploys the Street Time Machine app as a Python container. It
packages the app source in a ConfigMap and stores prepared recordings and
analysis data in a 2 GiB volume. The default uses temporary pod storage; choose
persistent storage if your cluster supports the requested PVC. The container
installs CPU dependencies from PyPI and does not request a GPU.

## Build the source bundle

Run from the repository root:

```sh
python deploy/street-time-machine/build_bundle.py
```

Copy this deployment directory to a machine with `kubectl` access to your
cluster. The generated `source.tgz.b64` bundle is local build output and is not
included in the repository.

## Prepare the app

Choose the namespace used by your cluster. By default, the scripts expect a
kubeconfig at `/config/<namespace>-k8s.yaml`; pass `--kubeconfig` when it is
stored elsewhere.

```sh
python3 deploy.py --phase prepare --namespace <namespace>
kubectl --kubeconfig=/config/<namespace>-k8s.yaml -n <namespace> \
  get pods -l app=street-time-machine
```

The default seed inputs are `~/stm` and `~/packd`. They must contain a prepared
camera configuration, recording, metadata, track cache, and a `segments.json`
file whose source URIs match the current catalogue results for that camera.
The seed command authenticates using one platform `*.config` file in
`/config`, whose `USERNAME`, `PASSWORD`, and `INGRESS_URL` fields are read at
runtime. Credentials are not embedded in the deployment manifests. Use
`--config`, `--prepared`, `--pack`, and `--camera` to select different paths or
inputs. Override `--kubeconfig` if the default path does not match your setup.

```sh
python3 seed.py
```

The readiness marker is written only after the transfer and refresh of saved
examples completes. Check the pod's `/health` route before publishing. The
seeded recording and analysis files remain in cluster storage and are not
source assets for this repository.

## Publish the app route

The example ingress host follows the VAST challenge cluster's
`video-lab-<namespace>.cosmos.vastdata.com` convention. Pass `--host` if your
cluster uses a different hostname. Set `--ingress-name` to the ingress
resource you intend to manage.

```sh
python3 deploy.py --phase publish --namespace <namespace> \
  --ingress-name street-time-machine --host <your-ingress-host>
```

The app is routed below `/app/`. The wrapper adds a readiness route, provenance
metadata, and a range-capable player for the original recording. New Ask
requests use the prepared track cache and the repository's rule-based question
engine; no LLM is added by this deployment.

## Storage and resource notes

With temporary storage, replacing the pod requires running `seed.py` again.
Persistent storage requires a functioning read-write-once PVC on the cluster.
Large synopsis or comparison preparation can exceed a small serving pod's
memory; build those results on a machine with sufficient memory, then seed the
prepared outputs. The example container requests 250 millicores and 512 MiB,
with limits of two CPUs and 2 GiB. Worker thread pools are capped at two to
keep CPU use bounded.

The dataset footage is not included. Supply recordings and prepared analyses
that you are authorized to use, and retain their applicable source credits and
licence terms.
