import control as ct
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, block_diag

from ross import MagneticBearingElement

# TODO: Ajustar a forma como essa classe é invocada no método _run_amb_tf_reduction_identification


class ReduceModel:
    """Build a reduced state-space model for a rotor with magnetic bearings.

    Parameters
    ----------
    rotor : Rotor
        Rotor model containing the magnetic bearings to be represented.
    speed : float, optional
        Rotor operating speed in rad/s. Default is 0.
    num_modes : int, optional
        Number of modes retained in the modal reduction. If set to ``-1``,
        all modes are retained. Default is ``-1``.

    Notes
    -----
    The magnetic bearings are removed from the rotor model during processing
    and represented through the reduced state-space model.

    Examples
    --------
    >>> from ross.bearings.magnetic.amb_models import rotor_example_amb_simple
    >>> rotor = rotor_example_amb_simple()
    >>> reducer = ReduceModel(rotor, speed=0.0, num_modes=4)
    """

    def __init__(self, rotor, speed=0, num_modes=-1):
        """Initialize the reduced model configuration.

        Parameters
        ----------
        rotor : Rotor
            Rotor model to be reduced.
        speed : float, optional
            Rotor operating speed in rad/s. Default is 0.
        num_modes : int, optional
            Number of modes retained in the modal reduction. If set to ``-1``,
            all modes are retained. Default is ``-1``.

        Returns
        -------
        None
            This method initializes the model state in place.

        Examples
        --------
        >>> from ross.bearings.magnetic.amb_models import rotor_example_amb_simple
        >>> rotor = rotor_example_amb_simple()
        >>> reducer = ReduceModel(rotor, speed=0.0, num_modes=4)
        """

        # Model configuration
        self.rotor = rotor  # Rotor model to be reduced
        self.speed = speed  # Operating speed of the rotor
        self.num_modes = num_modes  # Number of modes retained in the reduction

        # Rotor and magnetic bearing data
        self.n_dof = None  # Number of rotor degrees of freedom
        self.n_amb = None  # Number of active magnetic bearings
        self.node = None  # Nodes where magnetic bearings are installed
        self.k_s = None  # Magnetic bearing stiffness coefficients
        self.k_i = None  # Magnetic bearing current-force coefficients
        self.amb = None  # Magnetic bearing reference

        # State-space dimensions and state
        self.n_x = None  # Number of state variables
        self.n_u = None  # Number of control inputs
        self.x = None  # Current state vector

        # Original system models
        self.original_model_v = None  # Original vertical system model
        self.original_model_w = None  # Original horizontal system model
        self.G_v = None  # Vertical system transfer function
        self.G_w = None  # Horizontal system transfer function

        # Modal reduction matrices
        self.Phi = None  # Modal basis matrix
        self.M_m = None  # Reduced mass matrix
        self.C_m = None  # Reduced damping matrix
        self.K_m = None  # Reduced stiffness matrix

    def process_rotor(self):
        """Extract magnetic bearing data and initialize rotor dimensions.

        This method identifies the magnetic bearings in the rotor, stores their
        stiffness and current-force coefficients, removes them from the rotor
        model, and initializes the state-space dimensions.

        Returns
        -------
        None
            The rotor data and model attributes are updated in place.

        Notes
        -----
        Each magnetic bearing contributes two controlled directions, one for
        each transverse axis.

        Examples
        --------
        >>> from ross.bearings.magnetic.amb_models import rotor_example_amb_simple
        >>> reducer = ReduceModel(rotor_example_amb_simple())
        >>> reducer.process_rotor()
        >>> reducer.n_amb > 0
        True
        """

        self.n_dof = self.rotor.ndof

        # Identifying magnetic bearings
        magnetic_bearings = [
            brg
            for brg in self.rotor.bearing_elements
            if isinstance(brg, MagneticBearingElement)
        ]
        self.n_amb = len(magnetic_bearings)

        # Initialize lists
        self.k_i, self.k_s, self.node = [], [], []

        for amb in magnetic_bearings:
            # Constants
            self.k_i.extend([amb.ki, amb.ki])
            self.k_s.extend([amb.ks, amb.ks])
            self.node.extend([amb.n, amb.n])

        # Removing AMBs from model
        self.rotor.bearing_elements = [
            brg
            for brg in self.rotor.bearing_elements
            if not isinstance(brg, MagneticBearingElement)
        ]

        self.n_dof = self.rotor.ndof
        self.n_x = 2 * self.n_dof
        self.n_u = self.n_dof

        self.x = np.zeros((self.n_x, 1))

    def setup_modal_domain(self):
        """Compute the modal basis and reduced rotor matrices.

        The rotor mass, damping, and stiffness matrices are used to solve the
        generalized eigenvalue problem. The selected eigenvectors define the
        modal basis used to project the rotor matrices into the reduced domain.

        Returns
        -------
        None
            The modal basis and reduced matrices are stored as object attributes.

        Notes
        -----
        If ``num_modes`` is ``-1``, all available modes are retained and no
        modal reduction is applied.

        Examples
        --------
        >>> from ross.bearings.magnetic.amb_models import rotor_example_amb_simple
        >>> reducer = ReduceModel(rotor_example_amb_simple())
        >>> reducer.setup_modal_domain()
        """

        self.process_rotor()

        modal_reduction = True

        M = self.rotor.M(self.speed)
        C = self.rotor.C(self.speed)
        K = self.rotor.K(self.speed)

        eigenvalues, eigenvectors = eigh(K, M)

        if self.num_modes == -1:
            modal_reduction = False
            self.num_modes = len(eigenvalues)

        selected_eigenvalues = eigenvalues[: self.num_modes]

        omega_rad_s = np.sqrt(np.abs(selected_eigenvalues))
        freq_hz = omega_rad_s / (2 * np.pi)
        speed_rpm = freq_hz * 60

        if modal_reduction:
            print("\n" + 120 * "=")
            print(f"- Frequencies of the first {self.num_modes} modes")
            for i in range(self.num_modes):
                print(f"Mode {i + 1}: {freq_hz[i]:.2f} Hz | {speed_rpm[i]:.2f} RPM")
            print(120 * "=")

        self.Phi = eigenvectors[:, : self.num_modes]
        self.M_m = self.Phi.T @ M @ self.Phi
        self.C_m = self.Phi.T @ C @ self.Phi
        self.K_m = self.Phi.T @ K @ self.Phi

    def build_model(self):
        """Build the reduced magnetic-bearing state-space model.

        The method combines the reduced rotor dynamics with the magnetic
        bearing stiffness and current-force relationships to create a
        multi-input, multi-output state-space representation.

        Returns
        -------
        control.StateSpace
            Reduced state-space model with magnetic-bearing control inputs and
            displacement outputs.

        Notes
        -----
        The magnetic-bearing control axes are rotated by 45 degrees before
        being coupled to the reduced rotor model.

        Examples
        --------
        >>> from ross.bearings.magnetic.amb_models import rotor_example_amb_simple
        >>> reducer = ReduceModel(rotor_example_amb_simple())
        >>> reduced_model = reducer.build_model()
        >>> reduced_model.ninputs > 0 and reduced_model.noutputs > 0
        True
        """

        self.setup_modal_domain()

        theta = np.pi / 4  # Bearing orientation angle (45 degrees)
        n_controllers = 2 * self.n_amb

        Phi = self.Phi

        phi = np.zeros((self.n_dof, n_controllers))
        for i in range(self.n_amb):
            node = self.node[2 * i]
            phi[node * 6 + 0, 2 * i] = 1  # AMB i - X
            phi[node * 6 + 1, 2 * i + 1] = 1  # AMB i - Y

        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        r = np.array([[c_theta, s_theta], [-s_theta, c_theta]])
        R = block_diag(*[r for _ in range(self.n_amb)])

        K_x = np.diag(self.k_s)
        K_i = np.diag(self.k_i)

        N = self.n_dof
        n_c = n_controllers
        m = self.num_modes

        zeros = lambda rows, columns: np.zeros((rows, columns))

        H = np.block([np.eye(m), zeros(m, m)])
        phi_t = np.transpose(phi)

        M_m_I = np.linalg.inv(self.M_m)
        Phi_T = np.transpose(self.Phi)
        A = np.block([[zeros(m, m), np.eye(m)], [-M_m_I @ self.K_m, -M_m_I @ self.C_m]])
        B = np.block([[zeros(m, N)], [M_m_I @ Phi_T]])

        A_star = A + B @ phi @ K_x @ R @ phi_t @ Phi @ H
        B_star = B @ phi @ K_i
        C_star = R @ phi_t @ Phi @ H
        D_star = zeros(n_c, n_c)

        return ct.ss(A_star, B_star, C_star, D_star)


def plot_frequency_response(original_model, reduce_model):
    """Plot the frequency-response comparison of two system models.

    Parameters
    ----------
    original_model : control.StateSpace
        Original full-order system model.
    reduce_model : control.StateSpace
        Reduced-order system model to compare with the original model.

    Returns
    -------
    tuple of matplotlib.figure.Figure
        Magnitude and phase figures, returned in that order.

    Notes
    -----
    The function compares all 16 input-output channels and saves the figures
    as ``bode_magnitude_mimo.svg`` and ``bode_phase_mimo.svg``.

    Examples
    --------
    >>> from ross.bearings.magnetic.amb_models import rotor_example_amb_simple
    >>> reducer = ReduceModel(rotor_example_amb_simple())
    >>> reduced_model = reducer.build_model()
    >>> magnitude_figure, phase_figure = plot_frequency_response(
    ...     reduced_model, reduced_model
    ... )
    """

    omega = np.logspace(-2, 4, 1000)
    original_mag, original_phase, _ = ct.frequency_response(original_model, omega)
    reduce_mag, reduce_phase, _ = ct.frequency_response(reduce_model, omega)

    fig_magnitude, axes_magnitude = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
    fig_phase, axes_phase = plt.subplots(4, 4, figsize=(16, 12), sharex=True)

    for row in range(4):
        for column in range(4):
            original_mag_ij = original_mag[row, column, :]
            original_phase_ij = original_phase[row, column, :]
            reduce_mag_ij = reduce_mag[row, column, :]
            reduce_phase_ij = reduce_phase[row, column, :]

            original_mag_db = 20 * np.log10(
                np.maximum(original_mag_ij, np.finfo(float).tiny)
            )
            reduce_mag_db = 20 * np.log10(
                np.maximum(reduce_mag_ij, np.finfo(float).tiny)
            )
            original_phase_deg = np.rad2deg(np.unwrap(original_phase_ij))
            reduce_phase_deg = np.rad2deg(np.unwrap(reduce_phase_ij))

            magnitude_axis = axes_magnitude[row, column]
            phase_axis = axes_phase[row, column]

            magnitude_axis.semilogx(
                omega,
                original_mag_db,
                color="tab:blue",
                linestyle="-",
                label="Original",
            )
            magnitude_axis.semilogx(
                omega,
                reduce_mag_db,
                color="tab:orange",
                linestyle="--",
                label="Reduzido",
            )
            phase_axis.semilogx(
                omega,
                original_phase_deg,
                color="tab:blue",
                linestyle="-",
                label="Original",
            )
            phase_axis.semilogx(
                omega,
                reduce_phase_deg,
                color="tab:orange",
                linestyle="--",
                label="Reduzido",
            )

            magnitude_axis.set_title(f"G[{row + 1}, {column + 1}]")
            phase_axis.set_title(f"G[{row + 1}, {column + 1}]")
            magnitude_axis.grid(True, which="both", linestyle=":", alpha=0.7)
            phase_axis.grid(True, which="both", linestyle=":", alpha=0.7)

            if row == 3:
                magnitude_axis.set_xlabel("Frequência (rad/s)")
                phase_axis.set_xlabel("Frequência (rad/s)")
            if column == 0:
                magnitude_axis.set_ylabel("Magnitude (dB)")
                phase_axis.set_ylabel("Fase (graus)")

    axes_magnitude[0, 0].legend()
    axes_phase[0, 0].legend()
    fig_magnitude.suptitle("Diagrama de Bode: Magnitude", fontsize=16)
    fig_phase.suptitle("Diagrama de Bode: Fase", fontsize=16)
    fig_magnitude.tight_layout(rect=(0, 0, 1, 0.96))
    fig_phase.tight_layout(rect=(0, 0, 1, 0.96))
    fig_magnitude.savefig("bode_magnitude_mimo.svg", format="svg", bbox_inches="tight")
    fig_phase.savefig("bode_phase_mimo.svg", format="svg", bbox_inches="tight")

    return fig_magnitude, fig_phase
