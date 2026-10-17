"""Serve a private, local camera-review gallery."""
import argparse
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlparse

from build_gallery import load_catalog


class GalleryHandler(BaseHTTPRequestHandler):
    def send_file(self, path: Path, body: bool):
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        partial = False
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match or (not match[1] and not match[2]):
                self.send_error(416)
                return
            if not match[1]:
                count = int(match[2])
                start = max(0, size - count)
            else:
                start = int(match[1])
                end = min(int(match[2]) if match[2] else size - 1, size - 1)
            if start >= size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            partial = True
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if body:
            try:
                with path.open("rb") as stream:
                    stream.seek(start)
                    self.wfile.write(stream.read(end - start + 1))
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_HEAD(self):
        self.serve(False)

    def do_GET(self):
        self.serve(True)

    def serve(self, body):
        path = unquote(urlparse(self.path).path)
        if path == "/":
            target = self.gallery_dir / "index.html"
        elif path == "/catalog.json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            payload = json.dumps([{key: value for key, value in item.items() if key != "file"} for item in self.items]).encode()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if body:
                self.wfile.write(payload)
            return
        elif path.startswith("/video/"):
            item = self.by_id.get(path.removeprefix("/video/"))
            if item is None:
                self.send_error(404)
                return
            target = (self.media_root / item["file"]).resolve()
            try:
                target.relative_to(self.media_root)
            except ValueError:
                self.send_error(404)
                return
        else:
            self.send_error(404)
            return
        if not target.is_file():
            self.send_error(404)
            return
        self.send_file(target, body)

    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path(__file__).with_name("catalog.example.json"))
    parser.add_argument("--root", type=Path, required=True, help="directory containing local media")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18783)
    args = parser.parse_args()
    items = load_catalog(args.catalog, args.root)
    handler = type("ConfiguredGalleryHandler", (GalleryHandler,), {})
    handler.gallery_dir = Path(__file__).resolve().parent
    handler.media_root = args.root.resolve(strict=True)
    handler.items = items
    handler.by_id = {item["id"]: item for item in items}
    print(f"Gallery at http://{args.host}:{args.port}/ ({len(items)} clips)", flush=True)
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
