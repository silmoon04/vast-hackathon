import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

GALLERY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GALLERY))

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

build_gallery = load_module("build_gallery", GALLERY / "build_gallery.py")
server = load_module("gallery_server", GALLERY / "server.py")


class CatalogValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "media"
        self.root.mkdir()
        (self.root / "clip.mp4").write_bytes(b"0123456789")
        self.catalog = Path(self.temp.name) / "catalog.json"
        self.item = {
            "id": "a", "group": "g", "group_name": "G", "label": "A", "file": "clip.mp4",
            "duration": 10, "width": 1280, "height": 720,
            "preview_start": 0, "preview_end": 5,
        }

    def tearDown(self):
        self.temp.cleanup()

    def write(self, item=None):
        self.catalog.write_text(json.dumps([item or self.item], allow_nan=True), encoding="utf-8")

    def test_accepts_catalog_clip_under_root(self):
        self.write()
        self.assertEqual(build_gallery.load_catalog(self.catalog, self.root)[0]["id"], "a")

    def test_rejects_parent_traversal_and_absolute_paths(self):
        for value in ("../outside.mp4", str(self.root / "clip.mp4")):
            with self.subTest(value=value):
                item = {**self.item, "file": value}
                self.write(item)
                with self.assertRaises(ValueError):
                    build_gallery.load_catalog(self.catalog, self.root)

    def test_rejects_symlink_escape(self):
        outside = Path(self.temp.name) / "outside.mp4"
        outside.write_bytes(b"external")
        (self.root / "escape.mp4").symlink_to(outside)
        self.write({**self.item, "file": "escape.mp4"})
        with self.assertRaises(ValueError):
            build_gallery.load_catalog(self.catalog, self.root)

    def test_rejects_invalid_timing_dimensions_and_urls(self):
        invalid_items = [
            {**self.item, "duration": float("nan")},
            {**self.item, "width": 0},
            {**self.item, "height": float("inf")},
            {**self.item, "preview_start": -1},
            {**self.item, "preview_end": 11},
            {**self.item, "preview_start": 5, "preview_end": 5},
            {**self.item, "source_url": "javascript:alert(1)"},
            {**self.item, "license_url": "data:text/html,bad"},
        ]
        for item in invalid_items:
            with self.subTest(item=item):
                self.write(item)
                with self.assertRaises(ValueError):
                    build_gallery.load_catalog(self.catalog, self.root)

    def test_catalog_response_hides_local_file_path_and_supports_ranges(self):
        self.write()
        items = build_gallery.load_catalog(self.catalog, self.root)
        handler = type("TestHandler", (server.GalleryHandler,), {})
        handler.gallery_dir = GALLERY
        handler.media_root = self.root.resolve()
        handler.items = items
        handler.by_id = {item["id"]: item for item in items}
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_port}"
        try:
            with urlopen(base + "/catalog.json") as response:
                public_item = json.loads(response.read())[0]
            self.assertNotIn("file", public_item)
            request = Request(base + "/video/a", headers={"Range": "bytes=2-5"})
            with urlopen(request) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.headers["Content-Range"], "bytes 2-5/10")
                self.assertEqual(response.read(), b"2345")
            request = Request(base + "/video/a", headers={"Range": "bytes=40-50"})
            with self.assertRaises(HTTPError) as error:
                urlopen(request)
            self.assertEqual(error.exception.code, 416)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
