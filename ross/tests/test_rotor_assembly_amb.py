import numpy as np
import pytest
import control as ct
from numpy.testing import assert_allclose, assert_equal

from ross import (
    AmbSystemIdentificationResult,
    MagneticBearingElement,
    SensitivityResults,
)
from ross.bearings.magnetic.amb_models import (
    rotor_example_amb_simple,
    rotor_example_amb_general_controllers,
)
from ross.bearings.magnetic.amb_system_identification import (
    AmbSystemIdentification,
)
from ross.bearings.magnetic.system_identification import SystemIdentification
from ross.probe import Probe
from ross.units import Q_


def test_system_identification_fitness_uses_vector_norm():
    y_ref = np.array([1.0, 2.0, 4.0])
    y_pred = np.array([1.0, 3.0, 2.0])
    identification = SystemIdentification(np.zeros(3), y_ref, np.arange(3.0))

    expected = 100 * (
        1 - np.linalg.norm(y_ref - y_pred) / np.linalg.norm(y_ref - y_ref.mean())
    )

    assert_allclose(identification._fitness(y_pred), expected)


def test_system_identification_fitness_perfect_prediction_is_100():
    y_ref = np.array([1.0, 2.0, 4.0])
    identification = SystemIdentification(np.zeros(3), y_ref, np.arange(3.0))

    assert identification._fitness(y_ref) == 100.0


def test_system_identification_fitness_constant_reference_is_zero():
    identification = SystemIdentification(np.zeros(3), np.ones(3), np.arange(3.0))

    assert identification._fitness(np.zeros(3)) == 0.0


def test_amb_frequency_response_fitness_uses_complex_vector_norm():
    reference = np.array([1 + 1j, 2 + 3j, 4 - 2j])
    prediction = np.array([1 + 2j, 2 + 2j, 3 - 1j])
    expected = 100 * (
        1
        - np.linalg.norm(reference - prediction)
        / np.linalg.norm(reference - reference.mean())
    )

    assert_allclose(
        AmbSystemIdentification._frequency_response_fitness(reference, prediction),
        expected,
    )
    assert (
        AmbSystemIdentification._frequency_response_fitness(
            np.ones(3, dtype=complex), np.zeros(3, dtype=complex)
        )
        == 0.0
    )


def test_system_identification_identify_propagates_model_and_fitness(monkeypatch):
    identification = SystemIdentification(np.zeros(3), np.ones(3), np.arange(3.0))
    expected = (object(), 42.0, np.ones(3))
    monkeypatch.setattr(identification, "_identify_ss", lambda p: expected)

    assert identification.identify(method="ss", p=2) == expected


@pytest.mark.parametrize("method", ["de", "slsqp"])
def test_system_identification_transfer_methods_return_model_fitness_and_output(
    monkeypatch, method
):
    from types import SimpleNamespace

    import ross.bearings.magnetic.system_identification as system_identification

    t_ref = np.arange(10, dtype=float) * 0.01
    identification = SystemIdentification(
        np.zeros(t_ref.size), np.zeros(t_ref.size), t_ref
    )
    x = np.array([0.001, 0.001, 0.001, 0.001])

    if method == "de":

        def fake_differential_evolution(func, bounds, callback, **kwargs):
            return SimpleNamespace(x=x, fun=func(x))

        monkeypatch.setattr(
            system_identification,
            "differential_evolution",
            fake_differential_evolution,
        )
    else:

        def fake_minimize(fun, x0, method, callback, options):
            return SimpleNamespace(x=x, fun=fun(x))

        monkeypatch.setattr(system_identification, "minimize", fake_minimize)

    model, fitness, model_output = getattr(identification, f"_identify_{method}")()

    assert isinstance(model, ct.TransferFunction)
    assert isinstance(fitness, float)
    assert model_output.shape == t_ref.shape


def test_run_time_response_amb_values():
    rotor = rotor_example_amb_simple()
    t = np.arange(0, 2, 1e-6)
    F = np.zeros((len(t), rotor.ndof))
    speed = 0

    response = rotor.run_time_response(speed=speed, F=F, t=t, weight=True)

    expected_yout = np.array(
        [
            0.00000000e00,
            -1.35870226e-08,
            -5.43468162e-08,
            -1.22276734e-07,
            -2.17373160e-07,
            -3.39631510e-07,
            -4.89046215e-07,
            -6.65610691e-07,
            -8.69317307e-07,
            -1.10015734e-06,
        ]
    )

    expected_x_amb_0 = np.array(
        [
            0.00000000e00,
            2.52056841e-28,
            1.53697973e-27,
            4.36834784e-27,
            1.00165154e-26,
            2.31173210e-26,
            4.06316599e-26,
            5.63156133e-26,
            9.05077127e-26,
            1.14879669e-25,
        ]
    )

    expected_v_amb_0 = np.array(
        [
            0.00000000e00,
            -3.46710962e-12,
            -1.38683907e-11,
            -3.12037695e-11,
            -5.54731702e-11,
            -8.66765149e-11,
            -1.24813724e-10,
            -1.69884714e-10,
            -2.21889403e-10,
            -2.80827704e-10,
        ]
    )

    expected_F_y_amb_0 = np.array(
        [
            0.00000000e00,
            2.29979284e-06,
            9.19913968e-06,
            2.06979917e-05,
            3.67962986e-05,
            5.74940088e-05,
            8.27910692e-05,
            1.12687425e-04,
            1.47183022e-04,
            1.86277802e-04,
        ]
    )

    expected_F_v_amb_0 = np.array(
        [
            0.00000000e00,
            1.62619911e-06,
            6.50477405e-06,
            1.46356903e-05,
            2.60189123e-05,
            4.06544035e-05,
            5.85421265e-05,
            7.96820427e-05,
            1.04074113e-04,
            1.31718297e-04,
        ]
    )

    expected_I_v_amb_0 = np.array(
        [
            0.00000000e00,
            3.46710617e-07,
            1.38683769e-06,
            3.12037386e-06,
            5.54731155e-06,
            8.66764297e-06,
            1.24813601e-05,
            1.69884548e-05,
            2.21889187e-05,
            2.80827432e-05,
        ]
    )

    x_amb_0 = response.xout[0][:10, 0]
    v_amb_0 = response.xout[1][:10, 0]
    F_y_amb_0 = response.xout[2][:10, 1]
    F_v_amb_0 = response.xout[3][:10, 0]
    I_v_amb_0 = response.xout[4][:10, 0]
    yout = response.yout[:10, 5 * 6 + 1]

    assert_allclose(yout, expected_yout, atol=1e-5)
    assert_allclose(x_amb_0, expected_x_amb_0, atol=1e-5)
    assert_allclose(v_amb_0, expected_v_amb_0, atol=1e-5)
    assert_allclose(F_y_amb_0, expected_F_y_amb_0, atol=1e-5)
    assert_allclose(F_v_amb_0, expected_F_v_amb_0, atol=1e-5)
    assert_allclose(I_v_amb_0, expected_I_v_amb_0, atol=1e-5)

    fig1 = response.plot_amb_disps(axes=0)
    assert fig1 is not None

    fig2 = response.plot_amb_disps(axes=1)
    assert fig2 is not None

    fig3 = response.plot_amb_currents()
    assert fig3 is not None

    fig4 = response.plot_amb_forces(axes=0)
    assert fig4 is not None

    fig5 = response.plot_amb_forces(axes=1)
    assert fig5 is not None

    probe_disk_y = Probe(node=5, angle=Q_(90, "deg"), tag="Node 5 - Y")
    fig6 = response.plot_1d(probe=[probe_disk_y])
    assert fig6 is not None


def test_amb_controller():
    # Test for the magnetic_bearing_controller method.

    rot_speed = 1200
    dt = 0.001
    t = np.arange(0.0, 500 * dt, dt)
    unbalance_node = 27
    probe_node = 12

    rotor = rotor_example_amb_general_controllers()
    n = len(t)
    F = np.zeros((n, rotor.ndof))
    m_u = 0.010  # kg
    ex = 0.002  # m
    F0 = m_u * ex * rot_speed**2
    F[:, rotor.number_dof * unbalance_node + 0] = F0 * np.sin(rot_speed * t)
    F[:, rotor.number_dof * unbalance_node + 1] = F0 * np.cos(rot_speed * t)

    response = rotor.run_time_response(rot_speed, F, t, method="newmark")

    response_x = response.yout[:, rotor.number_dof * probe_node + 0]
    response_y = response.yout[:, rotor.number_dof * probe_node + 1]

    mse_x = 1 / n * np.sum(response_x**2)
    mse_y = 1 / n * np.sum(response_y**2)

    assert_allclose(mse_x, np.array(9.228097168398774e-10), rtol=1e-6, atol=1e-6)
    assert_allclose(mse_y, np.array(2.2135792430227363e-10), rtol=1e-6, atol=1e-6)


def test_amb_generic_controller():
    kp = 100.0
    ki = 0
    kd = 10.0
    n_f = 10_000

    s = MagneticBearingElement.s
    pid_controller = kp + ki / s + kd * s * (1 / (1 + (1 / n_f) * s))

    k_lead = 1
    T_lead = 0.5
    alpha_lead = 0.1
    lead_controller = k_lead * (T_lead * s + 1) / (alpha_lead * T_lead * s + 1)

    controller_transfer_function = pid_controller * lead_controller

    rot_speed = 1200
    dt = 0.001
    t = np.arange(0.0, 500 * dt, dt)
    unbalance_node = 27
    probe_node = 12

    rotor = rotor_example_amb_general_controllers(controller_transfer_function)
    n = len(t)
    F = np.zeros((n, rotor.ndof))
    m_u = 0.010  # kg
    ex = 0.002  # m
    F0 = m_u * ex * rot_speed**2
    F[:, rotor.number_dof * unbalance_node + 0] = F0 * np.sin(rot_speed * t)
    F[:, rotor.number_dof * unbalance_node + 1] = F0 * np.cos(rot_speed * t)

    response = rotor.run_time_response(rot_speed, F, t, method="newmark")

    response_x = response.yout[:, rotor.number_dof * probe_node + 0]
    response_y = response.yout[:, rotor.number_dof * probe_node + 1]

    mse_x = 1 / n * np.sum(response_x**2)
    mse_y = 1 / n * np.sum(response_y**2)

    assert_allclose(mse_x, np.array(7.934767106972457e-11), rtol=1e-6, atol=1e-6)
    assert_allclose(mse_y, np.array(3.6781959914042914e-11), rtol=1e-6, atol=1e-6)


def test_run_amb_sensitivity():
    """
    Tests the run_amb_sensitivity method for correctness of outputs and handling of various scenarios.
    """
    EXPECTED_SENSITIVITY_RESULTS = {
        "max_abs": {
            "Magnetic Bearing 0": {"x": 0.9915881235, "y": 0.9915881235},
            "Magnetic Bearing 1": {"x": 0.9880851953, "y": 0.9880851953},
        },
        "abs_slice": {
            "Magnetic Bearing 0": {
                "x": np.array(
                    [0.99158812, 0.99156866, 0.99153061, 0.99147841, 0.99142154]
                ),
                "y": np.array(
                    [0.99158812, 0.99156866, 0.99153061, 0.99147841, 0.99142154]
                ),
            },
            "Magnetic Bearing 1": {
                "x": np.array(
                    [0.9880852, 0.98805746, 0.98800146, 0.98792434, 0.98784035]
                ),
                "y": np.array(
                    [0.9880852, 0.98805746, 0.98800146, 0.98792434, 0.98784035]
                ),
            },
        },
        "phase_slice": {
            "Magnetic Bearing 0": {
                "x": np.array(
                    [
                        0.00000000e00,
                        8.77852477e-05,
                        1.59040274e-04,
                        2.11855244e-04,
                        2.44736262e-04,
                    ]
                ),
                "y": np.array(
                    [
                        0.00000000e00,
                        8.77852477e-05,
                        1.59040274e-04,
                        2.11855244e-04,
                        2.44736262e-04,
                    ]
                ),
            },
            "Magnetic Bearing 1": {
                "x": np.array(
                    [
                        0.00000000e00,
                        1.29420004e-04,
                        2.35610181e-04,
                        3.14075980e-04,
                        3.62979207e-04,
                    ]
                ),
                "y": np.array(
                    [
                        0.00000000e00,
                        1.29420004e-04,
                        2.35610181e-04,
                        3.14075980e-04,
                        3.62979207e-04,
                    ]
                ),
            },
        },
        "dofs": {
            "Magnetic Bearing 0": {"x": 72, "y": 73},
            "Magnetic Bearing 1": {"x": 258, "y": 259},
        },
        "time_results_slice": {
            "t": np.array([0.0, 0.0001, 0.0002, 0.0003, 0.0004]),
            "excitation": np.array(
                [
                    0.00000000e00,
                    6.67703996e-12,
                    1.42083014e-11,
                    2.27030685e-11,
                    3.22846065e-11,
                ]
            ),
            "disturbed": np.array(
                [
                    0.00000000e00,
                    6.67703996e-12,
                    1.42060807e-11,
                    2.26922938e-11,
                    3.22559336e-11,
                ]
            ),
            "sensor": np.array(
                [
                    0.00000000e00,
                    0.00000000e00,
                    -2.22067882e-15,
                    -1.07746729e-14,
                    -2.86728919e-14,
                ]
            ),
        },
        "frequencies_slice": np.array([0.0, 100.0, 200.0, 300.0, 400.0]),
    }

    r_tol = 0
    a_tol = 1e-8

    # Setup - run the analysis
    rotor = rotor_example_amb_general_controllers()
    results = rotor.run_amb_sensitivity(
        speed=0,
        t_max=1e-2,
        dt=1e-4,
        disturbance_amplitude=10e-6,
        disturbance_min_frequency=0.001,
        disturbance_max_frequency=150,
    )

    # Scenario 1: Default run verification
    # ------------------------------------
    assert isinstance(results, SensitivityResults)

    # Check types and shapes
    assert isinstance(results.sensitivities_frequencies, np.ndarray)
    assert isinstance(results.sensitivities_abs, dict)
    assert len(results.sensitivities_frequencies) == len(
        results.sensitivities_abs["Magnetic Bearing 0"]["x"]
    )

    # Check numerical values against golden values
    assert_allclose(
        results.sensitivities_frequencies[:5],
        EXPECTED_SENSITIVITY_RESULTS["frequencies_slice"],
        atol=a_tol,
        rtol=r_tol,
    )
    for amb_tag in results.max_abs_sensitivities:
        for axis in ["x", "y"]:
            assert_allclose(
                results.max_abs_sensitivities[amb_tag][axis],
                EXPECTED_SENSITIVITY_RESULTS["max_abs"][amb_tag][axis],
                atol=a_tol,
                rtol=r_tol,
            )
            assert_allclose(
                results.sensitivities_abs[amb_tag][axis][:5],
                EXPECTED_SENSITIVITY_RESULTS["abs_slice"][amb_tag][axis],
                atol=a_tol,
                rtol=r_tol,
            )
            assert_allclose(
                results.sensitivities_phase[amb_tag][axis][:5],
                EXPECTED_SENSITIVITY_RESULTS["phase_slice"][amb_tag][axis],
                atol=a_tol,
                rtol=r_tol,
            )

    assert_equal(results.sensitivity_compute_dofs, EXPECTED_SENSITIVITY_RESULTS["dofs"])

    time_results_amb_0_x = results.sensitivity_run_time_results["Magnetic Bearing 0"][
        "x"
    ]
    assert_allclose(
        results.sensitivity_run_time_results["t"][:5],
        EXPECTED_SENSITIVITY_RESULTS["time_results_slice"]["t"],
        atol=a_tol,
        rtol=r_tol,
    )
    assert_allclose(
        time_results_amb_0_x["excitation_signal"][:5],
        EXPECTED_SENSITIVITY_RESULTS["time_results_slice"]["excitation"],
        atol=a_tol,
        rtol=r_tol,
    )
    assert_allclose(
        time_results_amb_0_x["disturbed_signal"][:5],
        EXPECTED_SENSITIVITY_RESULTS["time_results_slice"]["disturbed"],
        atol=a_tol,
        rtol=r_tol,
    )
    assert_allclose(
        time_results_amb_0_x["sensor_signal"][:5],
        EXPECTED_SENSITIVITY_RESULTS["time_results_slice"]["sensor"],
        atol=a_tol,
        rtol=r_tol,
    )

    # Scenario 2: Test with `amb_tags` argument
    # -----------------------------------------
    results_tagged = rotor.run_amb_sensitivity(
        speed=1200, t_max=1e-2, dt=1e-4, amb_tags=["Magnetic Bearing 1"]
    )
    assert "Magnetic Bearing 1" in results_tagged.sensitivities
    assert "Magnetic Bearing 0" not in results_tagged.sensitivities
    assert len(results_tagged.sensitivities) == 1

    # Test for non-existent tag
    with pytest.raises(RuntimeError) as excinfo:
        rotor.run_amb_sensitivity(
            speed=1200, t_max=1e-2, dt=1e-4, amb_tags=["NonExistentAMB"]
        )
    assert "No Magnetic Bearing with the given tag was found" in str(excinfo.value)

    # Test for incorrect type for amb_tags
    with pytest.raises(ValueError) as excinfo:
        rotor.run_amb_sensitivity(
            speed=1200, t_max=1e-2, dt=1e-4, amb_tags="Magnetic Bearing 0"
        )
    assert "`amb_tags` must be a list of strings" in str(excinfo.value)

    # Scenario 3: Test with custom disturbance parameters
    # ----------------------------------------------------
    results_custom_freq = rotor.run_amb_sensitivity(
        speed=1200,
        t_max=1e-2,
        dt=1e-4,
        disturbance_min_frequency=10,
        disturbance_max_frequency=200,
    )
    # Check if max sensitivity differs, indicating parameters were used
    assert not np.allclose(
        results.max_abs_sensitivities["Magnetic Bearing 0"]["x"],
        results_custom_freq.max_abs_sensitivities["Magnetic Bearing 0"]["x"],
    )


def test_run_amb_system_identification():
    rotor = rotor_example_amb_simple()
    from ross.bearings.magnetic.amb_utils import get_ambs

    expected_keys = {
        f"{amb.tag} - {axis}"
        for amb in get_ambs(rotor)
        for axis in ("v", "w")
    }

    for method in ["de", "slsqp", "ss"]:
        result = rotor.run_amb_system_identification(
            method,
            dt=0.1,
            speed=0,
            time_to_converge=None,
            disturbance_changing_time=0.2,
            disturbance_steady_time=0.2,
            disturbance_n_steps=2,
            disturbance_step_amplitude=10e-6,
        )

        assert isinstance(result, AmbSystemIdentificationResult)
        assert result.method == method
        assert set(result.models) == expected_keys
        assert set(result.time_data) == expected_keys
        assert set(result.frequency_data) == expected_keys

        for key in expected_keys:
            model = result.models[key]
            data = result.time_data[key]
            fr_data = result.frequency_data[key]
            assert set(data) == {
                "t_ident",
                "d_ident",
                "current_ident",
                "disp_ident",
                "model_output_ident",
            }
            assert set(fr_data) == {
                "t_fr",
                "d_fr",
                "current_fr",
                "disp_fr",
                "f",
                "abs_fr",
                "phase_fr",
                "fr",
                "model_magnitude",
                "model_phase",
            }
            assert data["t_ident"].size == data["d_ident"].size
            assert data["t_ident"].size == data["model_output_ident"].size
            assert fr_data["t_fr"].size == fr_data["d_fr"].size
            assert fr_data["f"].size == fr_data["fr"].size
            for value in data.values():
                assert np.all(np.isfinite(value))
            for value in fr_data.values():
                assert np.all(np.isfinite(value))

            if method in ["de", "slsqp"]:
                assert isinstance(model, ct.TransferFunction)
                assert len(model.num[0][0]) > 0
                assert len(model.den[0][0]) > 0
                assert np.all(np.isfinite(model.num[0][0]))
                assert np.all(np.isfinite(model.den[0][0]))
            else:
                assert isinstance(model, ct.StateSpace)
                assert model.A.shape[0] == model.A.shape[1]
                assert model.B.shape[0] == model.A.shape[0]
                assert model.C.shape[1] == model.A.shape[0]
                assert np.all(np.isfinite(model.A))
                assert np.all(np.isfinite(model.B))
                assert np.all(np.isfinite(model.C))
                assert np.all(np.isfinite(model.D))

            assert data["t_ident"].size > 0
            assert np.all(np.diff(data["t_ident"]) >= 0)

    reduce_result = rotor.run_amb_system_identification("reduce", num_modes=10)
    assert reduce_result.time_data == {}
    assert reduce_result.frequency_data == {}


def test_run_amb_system_identification_rejects_unknown_method():
    rotor = rotor_example_amb_simple()

    with pytest.raises(ValueError, match="Unknown method 'unknown'"):
        rotor.run_amb_system_identification("unknown")


def test_system_identification_ss_accepts_single_and_multiple_p_values(monkeypatch):
    t_ref = np.arange(10, dtype=float)
    i_ref = np.ones_like(t_ref)
    y_ref = np.linspace(0.0, 1.0, t_ref.size)
    identification = SystemIdentification(i_ref, y_ref, t_ref)

    monkeypatch.setattr(
        identification,
        "_simulate_with_optimal_x0",
        lambda A, B: (np.zeros((t_ref.size, 2)), np.zeros(2)),
    )

    identification.identify(method="ss", p=2)
    assert identification.p == 2

    identification.identify(method="ss", p=[2, 3])
    assert identification.p in [2, 3]


def test_run_amb_system_identification_forwards_p(monkeypatch):
    rotor = rotor_example_amb_simple()
    identification = AmbSystemIdentification(rotor)
    forwarded = {}

    def fake_data_driven_identification(**kwargs):
        forwarded["p"] = kwargs["p"]
        kwargs["frequency_data"]["sample"] = {
            "f": np.array([1.0]),
            "abs_fr": np.array([2.0]),
            "phase_fr": np.array([3.0]),
        }

    monkeypatch.setattr(
        identification,
        "_run_amb_tf_data_driven_identification",
        fake_data_driven_identification,
    )

    result = identification.run_amb_system_identification("ss", p=[2, 4])

    assert forwarded["p"] == [2, 4]
    assert result.frequency_data["sample"]["f"].tolist() == [1.0]


def test_run_amb_system_identification_reduce_has_empty_domain_data(
    monkeypatch,
):
    rotor = rotor_example_amb_simple()
    identification = AmbSystemIdentification(rotor)
    forwarded = {}

    def fake_reduction(**kwargs):
        forwarded.update(kwargs)

    monkeypatch.setattr(
        identification, "_run_amb_tf_reduction_identification", fake_reduction
    )

    result = identification.run_amb_system_identification("reduce")

    assert result.time_data == {}
    assert result.frequency_data == {}


@pytest.mark.parametrize(
    "p, expected_p",
    [
        ([1, 2], [1, 2]),
        (3, [3, 3]),
        ([[10, 20, 30], [40, 50]], [[10, 20, 30], [40, 50]]),
    ],
)
def test_amb_tf_identification_uses_p_per_direction(monkeypatch, p, expected_p):
    rotor = rotor_example_amb_simple()
    identification = AmbSystemIdentification(rotor)
    forwarded = []

    class FakeSystemIdentification:
        def __init__(self, i_ref, y_ref, t_ref):
            self.t_ref = t_ref

        def identify(self, method, p):
            forwarded.append(p)
            return ct.tf([1.0], [1.0]), 12.5, np.zeros(self.t_ref.size)

    monkeypatch.setattr(
        "ross.bearings.magnetic.amb_system_identification.SystemIdentification",
        FakeSystemIdentification,
    )
    monkeypatch.setattr(
        "ross.bearings.magnetic.amb_system_identification.get_ident_disturbance",
        lambda *args, **kwargs: (np.arange(2.0), np.ones(2)),
    )
    monkeypatch.setattr(
        "ross.bearings.magnetic.amb_system_identification.compute_freq_resp",
        lambda *args: (
            np.array([10.0, 20.0]),
            np.array([1.0, 2.0]),
            np.array([0.1, 0.2]),
            np.array([1.0 + 0.0j, 2.0 + 0.0j]),
            None,
        ),
    )

    class FakeResponse:
        I = np.zeros((2, 4))
        v_amb = np.zeros((2, 4))

    monkeypatch.setattr(
        rotor, "run_time_response", lambda *args, **kwargs: FakeResponse()
    )

    result = identification.run_amb_system_identification(
        "ss", p=p, time_to_converge=1.0
    )

    assert forwarded == expected_p
    assert set(result.fit) == {"time", "freq"}
    assert set(result.fit["time"]) == set(result.models)
    assert set(result.fit["freq"]) == set(result.models)
    assert_equal(list(result.fit["time"].values()), [12.5] * len(result.models))
    assert set(result.time_data) == set(result.models)
    assert set(result.frequency_data) == set(result.models)
    for data in result.frequency_data.values():
        assert set(data) == {
            "t_fr",
            "d_fr",
            "current_fr",
            "disp_fr",
            "f",
            "abs_fr",
            "phase_fr",
            "fr",
            "model_magnitude",
            "model_phase",
        }
        assert_equal(data["f"], [10.0, 20.0])
        assert_equal(data["abs_fr"], [1.0, 2.0])
        assert_equal(data["phase_fr"], [0.1, 0.2])
        assert data["t_fr"].tolist() == [1.0]
        assert data["d_fr"].size == data["current_fr"].size == 1
        assert data["model_magnitude"].size == data["model_phase"].size == 2
    for data in result.time_data.values():
        assert data["t_ident"].tolist() == [1.0]
        assert data["d_ident"].size == data["current_ident"].size == 1
        assert data["model_output_ident"].size == 1

    key = next(iter(result.models))
    f = np.array([10.0, 20.0])
    model_magnitude, model_phase, _ = ct.frequency_response(
        result.models[key], 2 * np.pi * f
    )
    model_fr = np.squeeze(model_magnitude * np.exp(1j * model_phase))
    expected_fit = AmbSystemIdentification._frequency_response_fitness(
        np.array([1.0 + 0.0j, 2.0 + 0.0j]), model_fr
    )
    assert_allclose(result.fit["freq"][key], expected_fit)


def test_amb_tf_identification_generates_configurable_chirp(monkeypatch):
    rotor = rotor_example_amb_simple()
    identification = AmbSystemIdentification(rotor)
    chirp_calls = []

    class FakeSystemIdentification:
        def __init__(self, i_ref, y_ref, t_ref):
            pass

        def identify(self, method, p):
            return ct.tf([1.0], [1.0]), 0.0, np.zeros(4)

    monkeypatch.setattr(
        "ross.bearings.magnetic.amb_system_identification.SystemIdentification",
        FakeSystemIdentification,
    )
    monkeypatch.setattr(
        "ross.bearings.magnetic.amb_system_identification.get_ident_disturbance",
        lambda *args, **kwargs: (np.arange(4.0), np.ones(4)),
    )
    monkeypatch.setattr(
        "ross.bearings.magnetic.amb_system_identification.compute_freq_resp",
        lambda *args: (
            np.array([1.0, 2.0]),
            np.ones(2),
            np.zeros(2),
            np.ones(2, dtype=complex),
            np.ones(2),
        ),
    )

    class FakeResponse:
        I = np.zeros((4, 4))
        v_amb = np.zeros((4, 4))

    monkeypatch.setattr(
        rotor, "run_time_response", lambda *args, **kwargs: FakeResponse()
    )

    def fake_chirp(t, **kwargs):
        chirp_calls.append((t.copy(), kwargs))
        return np.ones_like(t)

    monkeypatch.setattr(
        "ross.bearings.magnetic.amb_system_identification.chirp", fake_chirp
    )

    identification.run_amb_system_identification(
        "ss",
        p=2,
        time_to_converge=None,
        disturbance_chirp_amplitude=2e-5,
        disturbance_chirp_min_frequency=0.2,
        disturbance_chirp_max_frequency=80,
    )

    assert len(chirp_calls) == 2
    t_chirp, kwargs = chirp_calls[0]
    assert_equal(t_chirp, np.arange(4.0))
    assert kwargs == {
        "f0": 0.2,
        "f1": 80,
        "t1": 3.0,
        "method": "logarithmic",
        "phi": -90,
    }


def test_amb_tf_identification_chirp_uses_default_parameters(monkeypatch):
    rotor = rotor_example_amb_simple()
    identification = AmbSystemIdentification(rotor)
    chirp_calls = []

    class FakeSystemIdentification:
        def __init__(self, i_ref, y_ref, t_ref):
            pass

        def identify(self, method, p):
            return ct.tf([1.0], [1.0]), 0.0, np.zeros(4)

    monkeypatch.setattr(
        "ross.bearings.magnetic.amb_system_identification.SystemIdentification",
        FakeSystemIdentification,
    )
    monkeypatch.setattr(
        "ross.bearings.magnetic.amb_system_identification.get_ident_disturbance",
        lambda *args, **kwargs: (np.arange(4.0), np.ones(4)),
    )
    monkeypatch.setattr(
        "ross.bearings.magnetic.amb_system_identification.compute_freq_resp",
        lambda *args: (
            np.array([1.0, 2.0]),
            np.ones(2),
            np.zeros(2),
            np.ones(2, dtype=complex),
            np.ones(2),
        ),
    )

    class FakeResponse:
        I = np.zeros((4, 4))
        v_amb = np.zeros((4, 4))

    monkeypatch.setattr(
        rotor, "run_time_response", lambda *args, **kwargs: FakeResponse()
    )
    monkeypatch.setattr(
        "ross.bearings.magnetic.amb_system_identification.chirp",
        lambda t, **kwargs: chirp_calls.append((t.copy(), kwargs)) or np.ones_like(t),
    )

    identification.run_amb_system_identification("ss", p=2, time_to_converge=None)

    assert len(chirp_calls) == 2
    assert chirp_calls[0][1] == {
        "f0": 1.0,
        "f1": 1000,
        "t1": 3.0,
        "method": "logarithmic",
        "phi": -90,
    }


@pytest.mark.parametrize("p", [[1], [1, 2, 3]])
def test_amb_tf_identification_rejects_invalid_p_sequence(p):
    rotor = rotor_example_amb_simple()

    with pytest.raises(ValueError, match="exactly 2 values"):
        rotor.run_amb_system_identification("ss", p=p)


@pytest.mark.parametrize("p", [[], [0], [1.5]])
def test_system_identification_ss_rejects_invalid_p_values(p):
    t_ref = np.arange(10, dtype=float)
    identification = SystemIdentification(
        np.ones_like(t_ref), np.ones_like(t_ref), t_ref
    )

    with pytest.raises(ValueError, match="'p'"):
        identification.identify(method="ss", p=p)


def test_amb_system_identification_plot_time_data():
    time = np.array([0.0, 0.1, 0.2])
    result = AmbSystemIdentificationResult(
        models={},
        fits={},
        method="ss",
        fit={"time": {"test": 91.25, "other": 82.5}, "freq": {}},
        time_data={
            "test": {
                "t_ident": time,
                "d_ident": np.array([1.0, 2.0, 3.0]),
                "current_ident": np.array([4.0, 5.0, 6.0]),
                "disp_ident": np.array([7.0, 8.0, 9.0]),
                "model_output_ident": np.array([10.0, 11.0, 12.0]),
            },
            "other": {
                "t_ident": time,
                "d_ident": np.array([1.0, 2.0, 3.0]),
                "current_ident": np.array([4.0, 5.0, 6.0]),
                "disp_ident": np.array([7.0, 8.0, 9.0]),
                "model_output_ident": np.array([10.0, 11.0, 12.0]),
            },
        },
    )

    fig = result.plot_time_data()

    assert len(fig.data) == 8
    assert_allclose(fig.data[0].x, time)
    assert_allclose(fig.data[0].y, [1.0, 2.0, 3.0])
    assert_allclose(fig.data[2].y, [7.0, 8.0, 9.0])
    assert_allclose(fig.data[3].y, [10.0, 11.0, 12.0])
    assert fig.data[0].legendgrouptitle.text == "Fitness: 91.25%"
    assert fig.data[4].legendgrouptitle.text == "Fitness: 82.50%"
    assert fig.layout.yaxis.title.text == "Disturbance (m)"
    assert fig.layout.yaxis3.title.text == "Displacement (m)"


def test_amb_system_identification_plot_frequency_data():
    result = AmbSystemIdentificationResult(
        models={},
        fits={},
        method="ss",
        fit={"time": {}, "freq": {"test": 76.5, "other": 63.25}},
        frequency_data={
            "test": {
                "t_fr": np.array([0.0, 1.0]),
                "d_fr": np.zeros(2),
                "current_fr": np.zeros(2),
                "disp_fr": np.zeros(2),
                "f": np.array([0.0, 1.0, 10.0]),
                "abs_fr": np.array([0.0, 2.0, 3.0]),
                "phase_fr": np.array([0.0, 0.2, 0.3]),
                "fr": np.array([0.0j, 2.0 + 0.1j, 3.0 + 0.2j]),
                "model_magnitude": np.array([0.0, 1.5, 2.5]),
                "model_phase": np.array([0.0, 0.1, 0.2]),
            },
            "other": {
                "t_fr": np.array([0.0, 1.0]),
                "d_fr": np.zeros(2),
                "current_fr": np.zeros(2),
                "disp_fr": np.zeros(2),
                "f": np.array([0.0, 1.0, 10.0]),
                "abs_fr": np.array([0.0, 2.0, 3.0]),
                "phase_fr": np.array([0.0, 0.2, 0.3]),
                "fr": np.array([0.0j, 2.0 + 0.1j, 3.0 + 0.2j]),
                "model_magnitude": np.array([0.0, 1.5, 2.5]),
                "model_phase": np.array([0.0, 0.1, 0.2]),
            },
        },
    )

    fig = result.plot_frequency_data(frequency_units="kHz")

    assert len(fig.data) == 8
    assert_allclose(fig.data[0].x, [0.001, 0.01])
    assert_allclose(fig.data[0].y, [2.0, 3.0])
    assert_allclose(fig.data[1].y, [1.5, 2.5])
    assert_allclose(fig.data[2].y, [0.2, 0.3])
    assert fig.data[0].legendgrouptitle.text == "Fitness: 76.50%"
    assert fig.data[4].legendgrouptitle.text == "Fitness: 63.25%"
    assert fig.layout.xaxis.type == "log"
    assert fig.layout.xaxis.matches == "x2"
    assert fig.layout.xaxis2.type == "log"
    assert fig.layout.yaxis.title.text == "Magnitude (m/A)"
    assert fig.layout.yaxis2.title.text == "Phase (rad)"


def test_amb_system_identification_plot_frequency_data_unwraps_phase():
    measured_phase = np.array([0.0, 3.0, 3.1, -3.1, -3.0, -2.9])
    model_phase = np.array([0.0, 2.9, 3.0, -3.0, -2.9, -2.8])
    result = AmbSystemIdentificationResult(
        models={},
        fits={},
        method="ss",
        frequency_data={
            "test": {
                "f": np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0]),
                "abs_fr": np.ones(6),
                "phase_fr": measured_phase.copy(),
                "model_magnitude": np.ones(6),
                "model_phase": model_phase.copy(),
            }
        },
    )

    fig = result.plot_frequency_data()

    assert_allclose(fig.data[2].x, [0.5, 1.0, 2.0, 3.0, 4.0])
    assert_allclose(fig.data[2].y, np.unwrap(measured_phase[1:]))
    assert_allclose(fig.data[3].y, np.unwrap(model_phase[1:]))
    assert np.all(np.abs(np.diff(fig.data[2].y)) < np.pi)
    assert np.all(np.abs(np.diff(fig.data[3].y)) < np.pi)
    assert_allclose(result.frequency_data["test"]["phase_fr"], measured_phase)
    assert_allclose(result.frequency_data["test"]["model_phase"], model_phase)


def test_amb_system_identification_plot_omits_missing_fitness():
    time = np.array([0.0, 0.1])
    time_result = AmbSystemIdentificationResult(
        models={},
        fits={},
        method="ss",
        fit={"time": {}, "freq": {}},
        time_data={
            "test": {
                "t_ident": time,
                "d_ident": np.zeros(2),
                "current_ident": np.zeros(2),
                "disp_ident": np.zeros(2),
                "model_output_ident": np.zeros(2),
            }
        },
    )
    frequency_result = AmbSystemIdentificationResult(
        models={},
        fits={},
        method="ss",
        fit={"time": {}, "freq": {}},
        frequency_data={
            "test": {
                "f": np.array([1.0, 2.0]),
                "abs_fr": np.ones(2),
                "phase_fr": np.zeros(2),
                "model_magnitude": np.ones(2),
                "model_phase": np.zeros(2),
            }
        },
    )

    assert time_result.plot_time_data().data[0].legendgrouptitle.text is None
    assert (
        frequency_result.plot_frequency_data().data[0].legendgrouptitle.text is None
    )


def test_amb_system_identification_plot_methods_warn_without_data():
    result = AmbSystemIdentificationResult(models={}, fits={}, method="reduce")

    with pytest.warns(UserWarning, match="No time-domain data"):
        assert result.plot_time_data() is None
    with pytest.warns(UserWarning, match="No frequency-domain data"):
        assert result.plot_frequency_data() is None
