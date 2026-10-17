"""Bundle the repository's STM package and UI for a flat Kubernetes ConfigMap."""
import base64
import io
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / 'street-time-machine'
ui_suffixes = {'.html', '.css', '.js', '.mjs', '.svg'}
ui_files = [path for path in (SOURCE / 'ui').rglob('*')
            if path.is_file() and path.suffix.lower() in ui_suffixes]
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
    files = sorted((SOURCE / 'stm').glob('*.py')) + sorted(ui_files)
    for path in files:
        archive.add(path, arcname=path.relative_to(SOURCE), recursive=False)
output = HERE / 'source.tgz.b64'
output.write_bytes(base64.b64encode(buffer.getvalue()))
print(f'Bundled {len(files)} source files into {output.name}: {output.stat().st_size} bytes')
