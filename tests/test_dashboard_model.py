from crystal_elastic_workbench.dashboard_model import build_dashboard_state
from crystal_elastic_workbench.examples import EXAMPLE_MATERIALS
from crystal_elastic_workbench.core import ElasticTensor
from crystal_elastic_workbench.stability import check_stability


def test_build_dashboard_state_formats_stable_summary_and_figures():
    example = EXAMPLE_MATERIALS["Si cubic"]
    tensor = ElasticTensor(
        example.matrix,
        crystal_system=example.crystal_system,
        unit=example.unit,
        material_name=example.name,
    )
    summary = tensor.polycrystalline_summary().as_dict()
    stability = check_stability(example.matrix, crystal_system=example.crystal_system)

    state = build_dashboard_state(summary, stability, figures_generated=True)

    assert state.status_text.startswith("Stable")
    assert state.metrics["B_H"].endswith("GPa")
    assert state.metrics["G_H"].endswith("GPa")
    assert state.metrics["E_H"].endswith("GPa")
    assert state.metrics["A_U"] == "0.244"
    assert "Pugh B/G=1.47" in state.anisotropy_text
    assert "Recommended model: Hill" in state.anisotropy_text
    assert state.figure_text == "Figures: 1D/2D/3D updated"


def test_build_dashboard_state_reports_incomplete_figures():
    example = EXAMPLE_MATERIALS["Si cubic"]
    tensor = ElasticTensor(example.matrix, crystal_system=example.crystal_system)
    summary = tensor.polycrystalline_summary().as_dict()
    stability = check_stability(example.matrix, crystal_system=example.crystal_system)

    state = build_dashboard_state(summary, stability, figures_generated=False)

    assert state.figure_text == "Figures: incomplete; check plot tabs"
