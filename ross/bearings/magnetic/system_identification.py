import time

import control as ct
import numpy as np
from scipy.optimize import differential_evolution, minimize

rng = np.random.default_rng(int(time.time()))


def get_disturbance(
    dt,
    changing_time=0.5,
    steady_time=2.0,
    n_steps=30,
    step_amplitude=50.0,
):
    """
    Generate a random, smoothly varying disturbance signal.

    The signal begins with a transient period at zero, followed by randomly
    signed levels. Consecutive levels are connected by a smooth fifth-order
    transition, and each level is held for a randomly varied duration.

    Parameters
    ----------
    dt : float
        Sampling interval in seconds. Must be positive.
    changing_time : float, optional
        Approximate duration of each transition between levels, in seconds.
        Defaults to 0.5.
    steady_time : float, optional
        Nominal duration of each constant-level segment, in seconds.
        Defaults to 2.0.
    n_steps : int, optional
        Number of randomly generated disturbance levels. Defaults to 30.
    step_amplitude : float, optional
        Amplitude of the disturbance levels. Defaults to 50.0.

    Returns
    -------
    t : ndarray
        Time vector in seconds.
    u : ndarray
        Disturbance values sampled at the times in ``t``.

    Notes
    -----
    The random generator is initialized when this module is imported. Calling
    this function therefore produces a different realization on each call.

    Examples
    --------
    >>> t, u = get_disturbance(
    ...     dt=0.01, changing_time=0.2, steady_time=0.5, n_steps=2
    ... )
    >>> t.shape == u.shape
    True
    """
    if np.round(changing_time / dt) <= 10:
        print(
            "Warning: changing_time is too short for the given dt. Consider increasing changing_time or decreasing dt."
        )

    n_transient_cycles = (
        5  # Number of initial cycles for stabilization (without disturbance)
    )
    magnitudes = step_amplitude * (2 * rng.random(n_steps + 1) - 1)
    signs = np.where(rng.random(n_steps + 1) < 0.5, -1, 1)
    values = magnitudes * signs
    values = np.concatenate((np.array(n_transient_cycles * [0.0]), values))

    n_changing = np.round(changing_time / dt).astype(int)
    n_changing = max(n_changing, 2)

    u = np.array([])

    for step in range(values.size - 1):
        curr_steady_time = steady_time * (0.9 * rng.random() + 0.1)

        n_steady = np.round(curr_steady_time / dt).astype(int)
        n_steady = max(n_steady, 2)
        n_steady_half = np.round(n_steady / 2).astype(int)

        u_0 = values[step] * np.ones(n_steady_half)
        u_2 = values[step + 1] * np.ones(n_steady - n_steady_half)

        t_changing = np.linspace(0, 1, n_changing)
        d_value = values[step + 1] - values[step]
        u_1 = (
            d_value * (t_changing**3 * (10.0 - 15.0 * t_changing + 6.0 * t_changing**2))
            + values[step]
        )

        u = np.concatenate((u, u_0, u_1, u_2))

    t = dt * np.arange(0, u.size)

    return t, u


def simulate_with_optimal_x0(A, B, t_ref, i_ref, y_ref):
    """
    Estimate initial conditions for a discrete state-space model.

    Computes the forced response and the two free responses associated with
    the unit initial states. The initial state that minimizes the least-
    squares displacement error is then estimated from these responses. The
    returned trajectory is the forced trajectory simulated from a zero
    initial state; the estimated initial state is returned separately.

    Parameters
    ----------
    A : ndarray
        Discrete-time state matrix with shape ``(2, 2)``.
    B : ndarray
        Discrete-time input matrix compatible with a scalar input.
    t_ref : array_like
        Reference time vector. Its length determines the number of samples.
    i_ref : array_like
        Reference input data (current), with one value per time sample.
    y_ref : array_like
        Reference output data (displacement), with one value per time sample.

    Returns
    -------
    x_pred : ndarray
        State trajectory obtained by applying ``i_ref`` with a zero initial
        state.
    x0_opt : ndarray
        Initial state vector that minimizes the least-squares displacement
        error for the reference data.

    Examples
    --------
    >>> A = np.array([[1.0, 0.1], [0.0, 1.0]])
    >>> B = np.array([[0.0], [0.1]])
    >>> t = np.arange(4, dtype=float) * 0.1
    >>> current = np.zeros(t.size)
    >>> displacement = np.zeros(t.size)
    >>> x_pred, x0_opt = simulate_with_optimal_x0(
    ...     A, B, t, current, displacement
    ... )
    >>> x_pred.shape, x0_opt.shape
    ((4, 2), (2,))
    """
    N = t_ref.size

    x_forced = np.zeros((N, 2))
    for k in range(N - 1):
        u_k = np.array([i_ref[k]])
        x_forced[k + 1, :] = (A @ x_forced[k, :].T + B @ u_k).flatten()

    x_free1 = np.zeros((N, 2))
    x_free1[0, :] = np.array([1, 0])
    for k in range(N - 1):
        x_free1[k + 1, :] = (A @ x_free1[k, :].T).flatten()

    x_free2 = np.zeros((N, 2))
    x_free2[0, :] = np.array([0, 1])
    for k in range(N - 1):
        x_free2[k + 1, :] = (A @ x_free2[k, :].T).flatten()

    M = np.column_stack((x_free1[:, 0], x_free2[:, 0]))
    b = y_ref - x_forced[:, 0]

    x0_opt, _, _, _ = np.linalg.lstsq(M, b, rcond=None)

    x_pred = np.zeros((N, 2))
    x_pred[0, :] = np.array([0, 0])

    for k in range(N - 1):
        x_k = x_pred[k, :]
        u_k = np.array([i_ref[k]])
        x_pred[k + 1, :] = (A @ x_k.T + B @ u_k).flatten()

    return x_pred, x0_opt


class SystemIdentification:
    """
    Identify second-order dynamic systems from input and output data.

    The class supports three identification algorithms:

    - ``"de"``: Differential Evolution optimization of transfer-function
      parameters.
    - ``"slsqp"``: SLSQP (Sequential Least Squares Programming)
      optimization of transfer-function parameters.
    - ``"ss"``: Multistep ordinary least squares identification of a
      discrete state-space model.

    Parameters
    ----------
    i_ref : array_like
        Reference input data, such as the bearing current.
    y_ref : array_like
        Reference output data, such as the measured displacement.
    t_ref : array_like
        Reference time vector corresponding to ``i_ref`` and ``y_ref``.

    Examples
    --------
    >>> t = np.arange(10, dtype=float) * 0.01
    >>> current = np.zeros(t.size)
    >>> displacement = np.zeros(t.size)
    >>> identification = SystemIdentification(current, displacement, t)
    """

    def __init__(self, i_ref, y_ref, t_ref):
        """
        Initialize a system-identification object.

        Parameters
        ----------
        i_ref : array_like
            Reference input data, such as the bearing current.
        y_ref : array_like
            Reference output data, such as the measured displacement.
        t_ref : array_like
            Reference time vector corresponding to ``i_ref`` and ``y_ref``.

        Notes
        -----
        The reference arrays are stored as provided and are used by all
        identification methods. They must have compatible lengths, and
        ``t_ref`` must contain sufficiently regular samples for the discrete
        state-space method.

        Examples
        --------
        >>> t = np.arange(5, dtype=float) * 0.01
        >>> identification = SystemIdentification(
        ...     np.zeros(5), np.zeros(5), t
        ... )
        """
        self.i_ref = i_ref  # Reference input data (current)
        self.y_ref = y_ref  # Reference output data (displacement)
        self.t_ref = t_ref  # Reference time vector

        self.function_call = 0  # Counter for cost function calls
        self._last_method = None  # Stores the last executed identification method

        self.result = None  # Result object from the optimization
        self.last_cost = 0.0  # Value of the last cost calculation

        self.de_generation = 0  # Current generation counter for DE
        self.prev_cost = np.inf  # Previous cost value for stagnation check
        self.stagnation_count = (
            0  # Counter for consecutive iterations with no improvement
        )
        self.max_stagnation = 10  # Maximum allowed iterations without improvement
        self.atol = 1e-4  # Absolute tolerance for convergence

        self.opt_iteration = 0  # Iteration counter for SLSQP

        self.p = 150  # Window size for State-Space identification
        self.v_ref = None  # Reference output derivative
        self.A = None  # Identified State-Space A matrix
        self.B = None  # Identified State-Space B matrix

    def _cost(self, X):
        """
        Evaluate the transfer-function identification objective.

        The objective is one half of the sum of squared errors between the
        reference output and the forced response of a second-order transfer
        function. Transfer functions with an unstable pole are rejected.

        Parameters
        ----------
        X : array_like
            Transfer-function parameters in the order ``[Kp, Tp1, Tp2, Tz]``.

        Returns
        -------
        cost : float
            Objective value. Returns ``np.inf`` if any pole is unstable.

        Notes
        -----
        The value is also stored in :attr:`last_cost`, and
        :attr:`function_call` is incremented on each invocation.

        Examples
        --------
        >>> t = np.arange(5, dtype=float) * 0.01
        >>> identification = SystemIdentification(
        ...     np.zeros(5), np.zeros(5), t
        ... )
        >>> cost = identification._cost([0.1, 0.01, 0.02, 0.01])
        >>> bool(np.isfinite(np.asarray(cost)).all())
        True
        """
        self.function_call += 1

        # X: [Kp, Tp1, Tp2, Tz]
        Kp, Tp1, Tp2, Tz = X[0], X[1], X[2], X[3]

        s = ct.tf("s")
        G = Kp * (1 + s * Tz) / ((1 + s * Tp1) * (1 + s * Tp2))
        poles = ct.poles(G)

        if np.any(poles > 0):
            cost = np.inf
        else:
            response = ct.forced_response(G, T=self.t_ref, U=self.i_ref)
            y = response.outputs

            err = y - self.y_ref
            cost = 0.5 * np.sum(err**2)

        self.last_cost = cost
        return cost

    def _de_callback(self, xk, convergence):
        """
        Report progress and detect stagnation during Differential Evolution.

        This callback is passed to :func:`scipy.optimize.differential_evolution`.
        It evaluates the current best candidate, prints progress, and requests
        termination after too many consecutive iterations without sufficient
        cost improvement.

        Parameters
        ----------
        xk : array_like
            Current best transfer-function parameter vector.
        convergence : float
            Convergence measure supplied by SciPy.

        Returns
        -------
        stop : bool
            ``True`` when stagnation has reached ``max_stagnation``;
            otherwise ``False``.

        Examples
        --------
        >>> t = np.arange(5, dtype=float) * 0.01
        >>> identification = SystemIdentification(
        ...     np.zeros(5), np.zeros(5), t
        ... )
        >>> from contextlib import redirect_stdout
        >>> from io import StringIO
        >>> with redirect_stdout(StringIO()):
        ...     stopped = identification._de_callback(
        ...         [0.1, 0.01, 0.02, 0.01], convergence=0.0
        ...     )
        >>> stopped
        False
        """
        self.de_generation += 1
        current_cost = self._cost(xk)
        X_str = np.array2string(np.asarray(xk), precision=6, floatmode="fixed")
        print(
            f"Generation {self.de_generation:03d} | cost={current_cost:.6f} | x_best={X_str} | conv={convergence}"
        )

        if abs(self.prev_cost - current_cost) < self.atol:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0

        self.prev_cost = current_cost

        if self.stagnation_count >= self.max_stagnation:
            print(
                f"Optimization stopped by stagnation after {self.stagnation_count} iterations."
            )
            return True

        return False

    def _slsqp_callback(self, xk):
        """
        Report progress during SLSQP optimization.

        This callback is passed to :func:`scipy.optimize.minimize` and prints
        the current parameter vector and most recently calculated cost.

        Parameters
        ----------
        xk : array_like
            Current transfer-function parameter vector.

        Examples
        --------
        >>> t = np.arange(5, dtype=float) * 0.01
        >>> identification = SystemIdentification(
        ...     np.zeros(5), np.zeros(5), t
        ... )
        >>> from contextlib import redirect_stdout
        >>> from io import StringIO
        >>> with redirect_stdout(StringIO()):
        ...     identification._slsqp_callback(
        ...         np.array([0.1, 0.01, 0.02, 0.01])
        ...     )
        >>> identification.opt_iteration
        1
        """
        self.opt_iteration += 1
        X_str = np.array2string(np.asarray(xk), precision=6, floatmode="fixed")
        print(
            f"Iteration {self.opt_iteration:04d} | x_best={X_str} | cost={self.last_cost:.4f}"
        )

    def _simulate_with_optimal_x0(self, A, B):
        """
        Estimate initial conditions for the identified state-space model.

        Delegates to :func:`simulate_with_optimal_x0` using this object's
        reference data.

        Parameters
        ----------
        A : ndarray
            Discrete-time state matrix.
        B : ndarray
            Discrete-time input matrix.

        Returns
        -------
        x_pred : ndarray
            State trajectory simulated from a zero initial state.
        x0_opt : ndarray
            Initial state vector estimated from the reference output.

        Examples
        --------
        >>> t = np.arange(4, dtype=float) * 0.1
        >>> identification = SystemIdentification(
        ...     np.zeros(4), np.zeros(4), t
        ... )
        >>> x_pred, x0_opt = identification._simulate_with_optimal_x0(
        ...     np.eye(2), np.zeros((2, 1))
        ... )
        """
        return simulate_with_optimal_x0(A, B, self.t_ref, self.i_ref, self.y_ref)

    def _identify_de(self):
        """
        Identify a transfer function using Differential Evolution.

        Optimizes the parameters ``[Kp, Tp1, Tp2, Tz]`` within the fixed
        bounds configured by this method and stores the SciPy optimization
        result in :attr:`result`.

        Returns
        -------
        control.TransferFunction
            Identified second-order transfer function with one zero.

        Examples
        --------
        >>> t = np.arange(10, dtype=float) * 0.01
        >>> identification = SystemIdentification(
        ...     np.zeros(10), np.zeros(10), t
        ... )
        >>> from contextlib import redirect_stdout
        >>> from io import StringIO
        >>> from types import SimpleNamespace
        >>> from unittest.mock import patch
        >>> def fake_differential_evolution(func, bounds, callback, **kwargs):
        ...     x = np.array([0.1, 0.01, 0.02, 0.01])
        ...     return SimpleNamespace(x=x, fun=func(x))
        >>> with patch(
        ...     "ross.bearings.magnetic.system_identification.differential_evolution",
        ...     fake_differential_evolution,
        ... ), redirect_stdout(StringIO()):
        ...     model = identification._identify_de()
        >>> isinstance(model, ct.TransferFunction)
        True
        """
        print("Starting identification using Differential Evolution (DE)...")
        bounds = [(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)]
        de_options = {
            "strategy": "rand1bin",
            "maxiter": 50,
            "popsize": 20,
            "tol": 1e-4,
            "atol": self.atol,
            "mutation": (0.5, 1.0),
            "recombination": 0.7,
            "polish": True,
            "disp": False,
            "updating": "deferred",
            "workers": -1,
            "seed": None,
        }

        tic = time.time()
        self.result = differential_evolution(
            func=self._cost,
            bounds=bounds,
            callback=self._de_callback,
            **de_options,
        )
        toc = time.time()

        print(f"Optimization finished in: {toc - tic:.2f} s")
        print(f"Best x: {self.result.x} | Best fit: {1 - self.result.fun}")
        Kp, Tp1, Tp2, Tz = self.result.x
        s = ct.tf("s")
        return Kp * (1 + s * Tz) / ((1 + s * Tp1) * (1 + s * Tp2))

    def _identify_slsqp(self):
        """
        Identify a transfer function using SLSQP optimization.

        Starts from the parameter vector ``[0.001, 0.001, 0.001, 0.001]``
        and stores the SciPy optimization result in :attr:`result`.

        Returns
        -------
        control.TransferFunction
            Identified second-order transfer function with one zero.

        Examples
        --------
        >>> t = np.arange(10, dtype=float) * 0.01
        >>> identification = SystemIdentification(
        ...     np.zeros(10), np.zeros(10), t
        ... )
        >>> from contextlib import redirect_stdout
        >>> from io import StringIO
        >>> from types import SimpleNamespace
        >>> from unittest.mock import patch
        >>> def fake_minimize(fun, x0, method, callback, options):
        ...     x = np.array([0.001, 0.001, 0.001, 0.001])
        ...     return SimpleNamespace(x=x, fun=fun(x))
        >>> with patch(
        ...     "ross.bearings.magnetic.system_identification.minimize",
        ...     fake_minimize,
        ... ), redirect_stdout(StringIO()):
        ...     model = identification._identify_slsqp()
        >>> isinstance(model, ct.TransferFunction)
        True
        """
        print("Starting identification using SLSQP...")
        x0 = np.array([0.001, 0.001, 0.001, 0.001])
        options = {"maxiter": 2000}

        tic = time.time()
        self.result = minimize(
            fun=self._cost,
            x0=x0,
            method="SLSQP",
            callback=self._slsqp_callback,
            options=options,
        )
        toc = time.time()

        print(f"Optimization finished in: {toc - tic:.2f} s")
        print(f"Best x: {self.result.x} | Best fit: {1 - self.result.fun}")
        Kp, Tp1, Tp2, Tz = self.result.x
        s = ct.tf("s")
        return Kp * (1 + s * Tz) / ((1 + s * Tp1) * (1 + s * Tp2))

    def _identify_ss(self, p_values):
        """
        Identify a discrete state-space model using multistep OLS.

        For each candidate memory length ``p``, estimates the state transition
        and input matrices from the reference displacement, its time
        derivative, and a window of input samples. The candidate with the
        lowest displacement error is selected.

        Parameters
        ----------
        p_values : int or iterable of int
            Positive candidate input-window lengths. Values greater than or
            equal to the number of samples are ignored.

        Returns
        -------
        control.StateSpace
            Identified discrete-time state-space model. Its output is the
            first state component, corresponding to displacement.

        Raises
        ------
        ValueError
            If ``p_values`` is empty, contains a non-positive or noninteger
            value, or contains no value smaller than the data length.

        Notes
        -----
        The selected matrices are also stored in :attr:`A`, :attr:`B`, and
        :attr:`p`; the numerical derivative of the reference output is stored
        in :attr:`v_ref`.

        Examples
        --------
        >>> t = np.arange(20, dtype=float) * 0.01
        >>> identification = SystemIdentification(
        ...     np.zeros(20), np.zeros(20), t
        ... )
        >>> from contextlib import redirect_stdout
        >>> from io import StringIO
        >>> with redirect_stdout(StringIO()):
        ...     model = identification._identify_ss([2])
        >>> isinstance(model, ct.StateSpace)
        True
        """
        if np.isscalar(p_values):
            p_values = [p_values]
        else:
            p_values = list(p_values)

        if not p_values:
            raise ValueError("'p' must contain at least one positive integer.")
        if any(
            isinstance(p_value, (bool, np.bool_))
            or not isinstance(p_value, (int, np.integer))
            or p_value <= 0
            for p_value in p_values
        ):
            raise ValueError("'p' must contain only positive integers.")

        print(
            f"Starting identification using State-Space (Multi-step OLS, p={p_values})..."
        )
        tic = time.time()
        self.v_ref = np.gradient(self.y_ref, self.t_ref)
        N = self.t_ref.size
        best_cost = np.inf
        best_p = None
        best_A = None
        best_B = None

        for p in p_values:
            if p >= N:
                continue

            Y_X = np.zeros((N - p, 2))
            Phi_X = np.zeros((N - p, 2 + p))
            for k in range(N - p):
                Y_X[k, :] = np.array([self.y_ref[k + p], self.v_ref[k + p]])
                x_k = [self.y_ref[k], self.v_ref[k]]
                u_window = self.i_ref[k : k + p].tolist()
                Phi_X[k, :] = np.array(x_k + u_window)

            Theta_X, _, _, _ = np.linalg.lstsq(Phi_X, Y_X, rcond=None)
            Theta_X_T = np.transpose(Theta_X)
            H_blocks = []
            for i in range(p):
                col_idx = 2 + p - 1 - i
                H_i = Theta_X_T[:, col_idx : col_idx + 1]
                H_blocks.append(H_i)

            B = H_blocks[0]
            M_left = np.hstack(H_blocks[:-1])
            M_right = np.hstack(H_blocks[1:])
            A = M_right @ np.linalg.pinv(M_left)
            x_pred, _ = self._simulate_with_optimal_x0(A, B)
            cost = 0.5 * np.sum((x_pred[:, 0] - self.y_ref) ** 2)
            print(f"p = {p} | cost = {cost:.4g}")

            if cost < best_cost:
                best_cost = cost
                best_p = p
                best_A = A
                best_B = B

        if best_p is None:
            raise ValueError("All values in 'p' must be smaller than the data length.")

        self.p = best_p
        self.A = best_A
        self.B = best_B
        toc = time.time()
        print(f"State-Space identification finished in: {toc - tic:.2f} s")
        print(f"Identification finished. Best p = {self.p} with cost = {best_cost:.4f}")

        dt = self.t_ref[1] - self.t_ref[0]
        C = np.array([[1, 0]])
        D = np.array([[0]])
        return ct.ss(self.A, self.B, C, D, dt=dt)

    def identify(self, method="de", **kwargs):
        """
        Run a selected system-identification algorithm.

        Executes one of the available identification methods using the
        reference data supplied at initialization.

        Parameters
        ----------
        method : str, optional
            Method to use: ``"de"`` for Differential Evolution, ``"slsqp"``
            for SLSQP optimization, or ``"ss"`` for state-space
            identification. Defaults to ``"de"``.
        **kwargs
            Additional method-specific options. For ``method="ss"``, ``p``
            may be an integer or iterable of candidate input-window lengths;
            it defaults to ``20``.

        Returns
        -------
        model : control.TransferFunction or control.StateSpace
            The identified system model.

        Raises
        ------
        ValueError
            If ``method`` is not one of ``"de"``, ``"slsqp"``, or ``"ss"``.

        Examples
        --------
        >>> t = np.arange(20, dtype=float) * 0.01
        >>> identification = SystemIdentification(
        ...     np.zeros(20), np.zeros(20), t
        ... )
        >>> from contextlib import redirect_stdout
        >>> from io import StringIO
        >>> with redirect_stdout(StringIO()):
        ...     model = identification.identify(method="ss", p=[2])
        >>> isinstance(model, ct.StateSpace)
        True
        """
        self._last_method = method.lower()
        self.function_call = 0
        identification_methods = {
            "de": self._identify_de,
            "slsqp": self._identify_slsqp,
            "ss": lambda: self._identify_ss(kwargs.get("p", 20)),
        }

        try:
            identify_method = identification_methods[self._last_method]
        except KeyError:
            raise ValueError(
                f"Unknown method '{method}'. Use 'de', 'slsqp' or 'ss'."
            ) from None
        return identify_method()
