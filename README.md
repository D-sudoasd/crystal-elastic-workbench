# Crystal Elastic Workbench

Crystal Elastic Workbench is a Python desktop tool for inspecting crystal elastic stiffness matrices (`Cij`).
It is aimed at research workflows where traceability matters: the program keeps the input matrix, unit,
crystal system, sampling parameters, plotting style, and export settings with the generated results.

The current priority is correctness and provenance first, then publication-quality visualization. The GUI
helps enter or import a `6 x 6` stiffness matrix, compute scalar and directional elastic properties, generate
1D/2D/3D plots, and export data/figures with sidecar manifests.

## What It Does

- Reads a full `6 x 6` Voigt stiffness matrix, normally in `GPa`.
- Provides input templates and examples for common crystal systems:
  `cubic`, `hexagonal`, `tetragonal`, `orthorhombic`, `trigonal`/`rhombohedral`,
  `monoclinic`, and `triclinic`.
- Checks matrix quality and stability:
  symmetry, invertibility, condition number, positive definiteness, and common Born criteria where available.
- Computes the compliance matrix `Sij`.
- Computes polycrystalline scalar parameters:
  `B_V`, `B_R`, `B_H`, `G_V`, `G_R`, `G_H`, Hill `E`, Hill `nu`, `B/G`,
  universal anisotropy index `A_U`, bulk/shear anisotropy percentages, and cubic Cauchy/Zener metrics.
- Computes directional properties:
  `E(n)`, linear compressibility `beta(n)`, transverse-mean `G(n,m)`, and transverse-mean `nu(n,m)`.
- Generates figures:
  1D plane curves, 1D high-symmetry/custom direction paths, 2D polar slices, 3D directional surfaces,
  rotating GIFs, and MP4 animations when `ffmpeg` is available.
- Exports analysis packages, sampled data, model tables, paper figure sets, and sidecar manifests.

## Installation

Use Python 3.11 on Windows:

```powershell
cd C:\Users\AORUS\Documents\Cij
py -3.11 -m pip install -r requirements.txt
```

For editable development:

```powershell
cd C:\Users\AORUS\Documents\Cij
py -3.11 -m pip install -e .[test]
```

Required runtime packages are listed in `requirements.txt`: `numpy`, `pandas`, `openpyxl`, `matplotlib`,
`pyvista`, `vtk`, `PySide6`, `imageio`, and `pytest`.

## Launching the GUI

From the repository:

```powershell
cd C:\Users\AORUS\Documents\Cij
py -3.11 -m crystal_elastic_workbench
```

After installation, the console script is also available:

```powershell
crystal-elastic-workbench
```

The batch launcher can be used on Windows:

```powershell
.\start_crystal_elastic_workbench.bat
```

## GUI Workflow

1. Choose a crystal system and material name.
2. Enter, paste, or import a `6 x 6` stiffness matrix.
3. Use `Analyze + Update Figures` to compute stability, scalar properties, model comparison, and default figures.
4. Inspect:
   - `Dashboard` for stability and key parameters.
   - `Results` for full scalar values and Voigt/Reuss/Hill/Geometric model comparison.
   - `1D`, `2D`, and `3D` tabs for directional plots.
5. Export data or figures from the relevant tab, or use `Export Full Package` / `Export Paper Figures`.

The matrix table has spreadsheet-style copy/paste support. Editing one off-diagonal cell updates its symmetric
mirror. Full `6 x 6` imports preserve the source matrix values so asymmetry can still be detected by analysis.

## Cij Input Convention

The fixed Voigt order is:

```text
[11, 22, 33, 23, 13, 12]
```

The stiffness matrix `C` and compliance matrix `S` use engineering shear strain:

```text
[e11, e22, e33, 2e23, 2e13, 2e12] = S [s11, s22, s33, s23, s13, s12]
```

Directional shear calculations apply physical stress tensors and convert the resulting engineering-strain
Voigt vector back to a symmetric strain tensor. This is intentional and avoids the common factor-of-two/four
mistake in shear terms.

## Example Data

Built-in GUI examples:

- `Al cubic`
- `Si cubic`
- `MgO cubic`

JSON examples are also provided:

- `examples/al_cubic.json`
- `examples/si_cubic.json`
- `examples/mgo_cubic.json`

These examples are for demonstration and regression testing. For publication or database comparison, verify
temperature, pressure, source convention, and units against your own reference.

## Calculation Notes

The Voigt/Reuss/Hill formulas use standard anisotropic elastic averages:

```text
B_V = (C11 + C22 + C33 + 2(C12 + C13 + C23)) / 9
G_V = (C11 + C22 + C33 - C12 - C13 - C23 + 3(C44 + C55 + C66)) / 15
B_R = 1 / (S11 + S22 + S33 + 2(S12 + S13 + S23))
G_R = 15 / (4(S11 + S22 + S33) - 4(S12 + S13 + S23) + 3(S44 + S55 + S66))
```

Hill values are arithmetic means of Voigt and Reuss. `G(n,m)` and `nu(n,m)` require `m` to be orthogonal to `n`.
For 3D shear and Poisson surfaces, the current GUI samples transverse directions around `n` and uses the mean
value by default. That is not the same as the strict maximum or minimum over all transverse directions.

## Exported Files

`Export Full Package` writes:

- `manifest.json`
- `stiffness_matrix.csv`
- `compliance_matrix.csv`
- `polycrystalline_summary.csv`
- `elastic_model_summary.csv`
- `elastic_model_summary.xlsx`
- `elastic_model_notes.json`
- `stability.json`
- `plane_xy_young.csv`
- `plane_xy_compressibility.csv`
- `surface_young.csv`
- `surface_compressibility.csv`
- `surface_shear.csv`
- `surface_poisson.csv`

Single-figure, animation, sampled-data, and model-table exports also write sidecar files named
`<filename>.manifest.json`. These manifests record the input `Cij`, unit, crystal system, export type,
plot style, palette, sampling parameters, and relevant render settings.

## 3D Rendering and Palettes

The preferred 3D path uses PyVista/VTK to render the surface and Matplotlib high-DPI composition for the title,
colorbar, and tick labels. This keeps the surface smooth while avoiding blurry PyVista text in saved PNGs and
GUI previews. If PyVista/VTK is unavailable, the GUI falls back to a simpler Matplotlib 3D surface.

Recommended 3D palettes are listed first in the GUI:

- Sequential: `Nature Surface` (default), `Nature Thermal`, `Blue-Gold`, `Teal-Amber`, `Deep Ocean`
- Diverging: `Blue-White-Red`, `Purple-Green`, `Brown-Blue`
- Print-safe: `Graphite`, `Gray Print`
- Compatibility palettes: `Cividis`, `Viridis Refined`

Use sequential palettes for modulus or intensity surfaces. Use diverging palettes only when the quantity has a
meaningful center, signed contrast, or difference field.

## Testing

Run the full suite:

```powershell
cd C:\Users\AORUS\Documents\Cij
py -3.11 -m pytest -q
```

Focused checks:

```powershell
py -3.11 -m pytest tests/test_elastic_core.py tests/test_sampling_export.py -q
py -3.11 -m pytest tests/test_gui_smoke.py tests/test_figure_export.py tests/test_paper_export.py -q
py -3.11 -m pytest tests/test_render3d.py tests/test_animation_export.py -q
```

The tests cover isotropic-cubic analytical checks, engineering-shear conventions, stability checks, sampling,
manifest writing, GUI smoke behavior, figure export, 3D rendering metadata, palette defaults, and animation
option forwarding.

## Known Limits

- The program can reduce tested risk but cannot prove that arbitrary user-provided `Cij` values are scientifically
  correct. Always verify units, source convention, temperature, pressure, and symmetry before publication.
- Trigonal and monoclinic templates follow common conventions; papers and software may use different axis/sign
  conventions. Check the original source before entering constants.
- Monoclinic and triclinic systems currently rely mainly on positive definiteness rather than compact Born-rule
  shortcuts.
- 3D shear/Poisson surfaces use transverse means by default, not strict transverse extrema.
- Higher 3D sampling grids can be slow for shear/Poisson because each direction requires a transverse scan.
- MP4 export requires a working local `ffmpeg`; GIF export is the safer fallback.
- Generated preview files and Python caches are intentionally ignored by Git.

## Merge-Readiness Checklist

Before merging into `main`, verify:

- `git status --short --branch` shows only intended source, test, doc, and configuration changes.
- `py -3.11 -m pytest -q` passes.
- README instructions still match the available commands and GUI behavior.
- Export manifests contain enough provenance to reproduce the displayed figure or sampled data.
- No generated caches, `outputs/`, or temporary files are staged.
