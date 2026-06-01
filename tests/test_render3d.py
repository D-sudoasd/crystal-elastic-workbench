import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from crystal_elastic_workbench.core import ElasticTensor
from crystal_elastic_workbench.render3d import (
    PyVistaUnavailableError,
    Render3DOptions,
    pyvista_status,
    render_surface_image,
    render_surface_png,
)
from crystal_elastic_workbench.sampling import sample_sphere


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


def test_render3d_defaults_match_high_quality_white_backend():
    options = Render3DOptions()

    assert options.theme_name == "Nature White"
    assert options.palette_name == "Nature Surface"
    assert options.window_size == (2400, 2000)
    assert options.parallel_projection is True
    assert options.smooth_shading is True
    assert options.transparent_background is False
    assert options.compose_annotations is True
    assert options.title_font_size == 18
    assert options.label_font_size == 12
    assert options.colorbar_title_size == 12
    assert options.colorbar_tick_size == 10
    assert options.lighting_intensity == pytest.approx(1.0)
    assert options.surface_subdivision == 1
    assert options.ambient == pytest.approx(0.28)
    assert options.diffuse == pytest.approx(0.74)
    assert options.specular == pytest.approx(0.32)
    assert options.specular_power == pytest.approx(28.0)


def test_render3d_style_parameters_include_palette_and_annotation_metadata():
    from crystal_elastic_workbench.render3d import render3d_style_parameters

    parameters = render3d_style_parameters(Render3DOptions())

    assert parameters["palette"] == "Nature Surface"
    assert parameters["palette_category"] == "sequential"
    assert parameters["compose_annotations"] is True
    assert parameters["annotation_backend"] == "matplotlib"
    assert parameters["title_font_size"] == 18
    assert parameters["label_font_size"] == 12
    assert parameters["colorbar_title_size"] == 12
    assert parameters["colorbar_tick_size"] == 10


def test_render_surface_image_returns_nonblank_preview_or_clear_unavailable_message():
    tensor = ElasticTensor(isotropic_cubic_matrix(), crystal_system="cubic")
    surface = sample_sphere(tensor, property_name="young", theta_count=5, phi_count=9)
    status = pyvista_status()

    if not status.available:
        with pytest.raises(PyVistaUnavailableError, match="PyVista/VTK"):
            render_surface_image(surface, options=Render3DOptions(window_size=(320, 260)))
        return

    image = render_surface_image(surface, options=Render3DOptions(window_size=(320, 260)))

    assert image.ndim == 3
    assert image.shape[0] == 260
    assert image.shape[1] == 320
    assert image.shape[2] in {3, 4}
    assert float(np.std(image)) > 0.0


def test_pyvista_png_render_or_clear_unavailable_message(tmp_path):
    tensor = ElasticTensor(isotropic_cubic_matrix(), crystal_system="cubic")
    surface = sample_sphere(tensor, property_name="young", theta_count=5, phi_count=9)
    output = tmp_path / "surface.png"
    status = pyvista_status()

    if not status.available:
        with pytest.raises(PyVistaUnavailableError, match="PyVista/VTK"):
            render_surface_png(surface, output, options=Render3DOptions(window_size=(320, 260)))
        assert not output.exists()
        return

    render_surface_png(surface, output, options=Render3DOptions(window_size=(320, 260)))

    assert output.stat().st_size > 1000
