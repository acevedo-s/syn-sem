import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from codex_helpers.helpers import convert_snapshot_to_safetensors


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python convert_checkpoint_to_safetensors.py <snapshot_dir>")

    convert_snapshot_to_safetensors(Path(sys.argv[1]).resolve())
