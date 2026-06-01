import matplotlib
import numpy as np

matplotlib.use("Agg")

from crystal_elastic_workbench.core import ElasticTensor
from crystal_elastic_workbench.sampling import sample_plane, sample_sphere
from crystal_elastic_workbench.visualization import (
    export_rotating_gif,
    export_rotating_mp4,
    plot_direction_path,
    plot_directional_surface,
    plot_plane_slice,
)


def isotropic_cubic_matrix(bulk_gpa: float = 160.0, shear_gpa: float = 80.0) -> np.ndarray:
    c11 = bulk_gpa + 4.0 * shear_gpa / 3.0
    c12 = bulk_gpa - 2.0 * shear_gpa / 3.0
    c44 = shear_gpa
    return np.array(
        [
            [c11, c12, c12, 0.0, 0.0, 0.0],
            [c12, c11, c12, 0.0, 0.0, 0.0],
            [c12, c12, c11, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, c44, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, c44, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, c44],
        ],
        dtype=float,
    )


def test_plot_exports_static_png_files(tmp_path):
    tensor = ElasticTensor(isotropic_cubic_matrix(), crystal_system="cubic")
    plane = sample_plane(tensor, property_name="young", plane="xy", angle_count=37)
    surface = sample_sphere(tensor, property_name="young", theta_count=7, phi_count=13)

    plane_fig = plot_plane_slice(plane)
    surface_fig = plot_directional_surface(surface)

    plane_path = tmp_path / "plane.png"
    surface_path = tmp_path / "surface.png"
    plane_fig.savefig(plane_path, dpi=120)
    surface_fig.savefig(surface_path, dpi=120)

    assert plane_path.stat().st_size > 1000
    assert surface_path.stat().st_size > 1000


def test_direction_path_uses_compact_publication_styling():
    from crystal_elastic_workbench.sampling import sample_direction_path

    tensor = ElasticTensor(isotropic_cubic_matrix(), crystal_system="cubic")
    path = sample_direction_path(
        tensor,
        property_name="young",
        points=[
            ("[100]", [1, 0, 0]),
            ("[110]", [1, 1, 0]),
            ("[111]", [1, 1, 1]),
        ],
        points_per_segment=5,
    )

    fig = plot_direction_path(path, theme_name="Nature White", palette_name="Okabe-Ito")
    ax = fig.axes[0]

    assert ax.get_title()
    assert ax.get_xlabel() == "Direction path"
    assert [label.get_text() for label in ax.get_xticklabels()] == ["[100]", "[110]", "[111]"]
    assert fig.dpi == 150
    assert ax.lines[0].get_linewidth() <= 1.5


def test_export_rotating_gif_writes_animation(tmp_path):
    tensor = ElasticTensor(isotropic_cubic_matrix(), crystal_system="cubic")
    surface = sample_sphere(tensor, property_name="young", theta_count=7, phi_count=13)
    gif_path = tmp_path / "rotation.gif"

    export_rotating_gif(surface, gif_path, frames=5, dpi=80)

    assert gif_path.stat().st_size > 1000


def test_export_rotating_mp4_prefers_pyvista_backend(tmp_path, monkeypatch):
    tensor = ElasticTensor(isotropic_cubic_matrix(), crystal_system="cubic")
    surface = sample_sphere(tensor, property_name="young", theta_count=7, phi_count=13)
    seen = {}

    def fake_render(surface_arg, output_path, **kwargs):
        seen["surface"] = surface_arg
        seen["kwargs"] = kwargs
        output_path.write_bytes(b"mp4 bytes")
        return output_path

    monkeypatch.setattr("crystal_elastic_workbench.visualization.render_surface_mp4", fake_render)
    mp4_path = tmp_path / "rotation.mp4"

    export_rotating_mp4(surface, mp4_path, frames=7, dpi=80, surface_subdivision=2, specular=0.4)

    assert seen["surface"] is surface
    assert seen["kwargs"]["frames"] == 7
    assert seen["kwargs"]["options"].surface_subdivision == 2
    assert seen["kwargs"]["options"].specular == 0.4
    assert mp4_path.read_bytes() == b"mp4 bytes"
