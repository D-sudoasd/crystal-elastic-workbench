from pathlib import Path


def test_readme_documents_merge_gate_relevant_workflows():
    text = Path("README.md").read_text(encoding="utf-8")

    required_sections = [
        "## Installation",
        "## Launching the GUI",
        "## GUI Workflow",
        "## Cij Input Convention",
        "## Exported Files",
        "## 3D Rendering and Palettes",
        "## Testing",
        "## Known Limits",
        "## Merge-Readiness Checklist",
    ]
    for section in required_sections:
        assert section in text
    assert "py -3.11 -m pytest -q" in text
    assert "Voigt order" in text
    assert "sidecar" in text
    assert "\ufffd" not in text
