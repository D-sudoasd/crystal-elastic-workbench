"""Subprocess entrypoint for isolated PyVista 3D rendering."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

from crystal_elastic_workbench.render3d import render_surface_png


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: render3d_worker <payload.pkl> <output.png>", file=sys.stderr)
        return 2
    payload_path = Path(args[0])
    output_path = Path(args[1])
    with payload_path.open("rb") as handle:
        payload = pickle.load(handle)
    render_surface_png(payload["surface"], output_path, options=payload["options"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
