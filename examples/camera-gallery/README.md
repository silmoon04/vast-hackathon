# Local camera gallery

A dependency-free, local web gallery for reviewing several camera recordings on a shared source-time clock. Pick a scene, preview its clips, compare up to five views together, keep selections in browser storage, and export the selected catalog metadata as JSON.

## Run

Use Python 3.10 or newer. Put your own video files under a media directory and create a JSON catalog based on `catalog.example.json`. Each `file` is relative to the media root; `id` values must be unique. Groups represent synchronized scenes. The example paths are placeholders and must be replaced with files you own or are authorized to review.

```powershell
python examples/camera-gallery/build_gallery.py --root C:\path\to\media --catalog C:\path\to\catalog.json
python examples/camera-gallery/server.py --root C:\path\to\media --catalog C:\path\to\catalog.json --port 18783
```

Open `http://127.0.0.1:18783/`. The server binds to loopback by default. Change `--host` only if you intend to make the gallery available on another interface. Video byte ranges are supported for browser seeking. Only catalog-listed media is served, and resolved paths must remain inside `--root`, including when symlinks are present.

## Catalog fields

Required: `id`, `group`, `group_name`, `label`, and `file`. The interface also uses `duration`, `width`, `height`, `preview_start`, `preview_end`, and `synthetic`; provide these for useful labels and previews. Add `credit`, `source_url`, `license`, and `license_url` when applicable. `camera` is optional. Times are seconds from the start of each source recording; only group clips with a common timeline for synchronized comparison.

The app does not upload media. “Keep” selections remain in local browser storage; export downloads the selected metadata and omits local file paths.

## Check a catalog

```powershell
python examples/camera-gallery/build_gallery.py --root C:\path\to\media --catalog C:\path\to\catalog.json
```

No third-party Python packages are needed.
