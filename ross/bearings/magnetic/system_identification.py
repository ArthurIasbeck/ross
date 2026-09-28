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
    if np.round(changing_time / dt) <= 10:
        print(
            "Warning: changing_time is too short for the given dt. Consider increasing changing_time or decreasing dt."
        )

    n_transient_cicles = (
        5  # Número de ciclos iniciais para estabilização (sem perturbação)
    )
    magnitudes = step_amplitude * (0.5 + rng.random(n_steps + 1))
    signs = np.where(rng.random(n_steps + 1) < 0.5, -1, 1)
    values = magnitudes * signs
    values = np.concatenate((np.array(n_transient_cicles * [0.0]), values))

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
    Simulate state-space system with optimal initial conditions.

    Estimates the initial state vector that minimizes error for a given
    state-space model (A, B) and returns the simulated output.

    Parameters
    ----------
    A : ndarray
        System state matrix.
    B : ndarray
        Input matrix.
    t_ref : array_like
        Reference time vector.
    i_ref : array_like
        Reference input data (current).
    y_ref : array_like
        Reference output data (displacement).

    Returns
    -------
    x_pred : ndarray
        Predicted state trajectory.
    x0_opt : ndarray
        Optimal initial state vector.
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
    # x_pred[0, :] = x0_opt
    x_pred[0, :] = np.array([0, 0])

    for k in range(N - 1):
        x_k = x_pred[k, :]
        u_k = np.array([i_ref[k]])
        x_pred[k + 1, :] = (A @ x_k.T + B @ u_k).flatten()

    return x_pred, x0_opt


class SystemIdentification:
    """
    Unified class for parameter identification of second-order dynamic systems.

    Supports three identification algorithms:
    - 'de': Differential Evolution to adjust Transfer Function parameters.
    - 'slsqp': Classic SLSQP optimization to adjust Transfer Function parameters.
    - 'ss': Least Squares identification of a Discrete State-Space model.

    Parameters
    ----------
    i_ref : array_like
        Reference input data (current).
    y_ref : array_like
        Reference output data (displacement).
    t_ref : array_like
        Reference time vector.
    """

    def __init__(self, i_ref, y_ref, t_ref):
        """
        Initialize the AmbSystemIdentification class.

        Parameters
        ----------
        i_ref : array_like
            Reference input data (current).
        y_ref : array_like
            Reference output data (displacement).
        t_ref : array_like
            Reference time vector.
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
        Calculate the cost function for Transfer Function identification.

        Computes the sum of squared errors between the reference output
        and the output of a second-order transfer function defined by parameters X.

        Parameters
        ----------
        X : array_like
            Parameters for the transfer function: [Kp, Tp1, Tp2, Tz].

        Returns
        -------
        cost : float
            The computed cost value. Returns np.inf if poles are unstable.
        """
        self.function_call += 1

        # X = [Kp, Tp1, Tp2, Tz]
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
        Callback function for Differential Evolution optimization.

        Prints generation progress and checks for optimization stagnation.

        Parameters
        ----------
        xk : array_like
            The current best parameter vector.
        convergence : float
            The convergence measure.

        Returns
        -------
        stop : bool
            True if optimization should stop, False otherwise.
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
        Callback function for SLSQP optimization.

        Prints iteration progress during SLSQP optimization.

        Parameters
        ----------
        xk : array_like
            The current parameter vector.
        """
        self.opt_iteration += 1
        X_str = np.array2string(np.asarray(xk), precision=6, floatmode="fixed")
        print(
            f"Iteration {self.opt_iteration:04d} | x_best={X_str} | cost={self.last_cost:.4f}"
        )

    def _simulate_with_optimal_x0(self, A, B):
        """
        Simulate state-space system with optimal initial conditions.

        Estimates the initial state vector that minimizes error for a given
        state-space model (A, B) and returns the simulated output.

        Parameters
        ----------
        A : ndarray
            System state matrix.
        B : ndarray
            Input matrix.

        Returns
        -------
        x_pred : ndarray
            Predicted state trajectory.
        x0_opt : ndarray
            Optimal initial state vector.
        """
        return simulate_with_optimal_x0(A, B, self.t_ref, self.i_ref, self.y_ref)

    def identify(self, method="de", **kwargs):
        """
        Run the identification algorithm.

        Executes the chosen identification method ('de', 'slsqp', or 'ss')
        to find the optimal parameters for the system model based on the
        provided reference data.

        Parameters
        ----------
        method : str, optional
            The identification method to use. Options are 'de' (Differential
            Evolution), 'slsqp' (SLSQP optimization), or 'ss' (State-Space
            identification). Defaults to 'de'.
        p : int or array_like of int, optional
            State-space prediction window length or candidate window lengths.
            The window determines how many subsequent input samples are
            considered by the model when making a prediction. Larger values
            consider more inputs. This parameter is used only when
            ``method='ss'`` and defaults to 20.

        Returns
        -------
        model : control.TransferFunction or control.StateSpace
            The identified system model.

        Examples
        --------
        >>> i_ref = np.array([0.1, 0.2, 0.1])
        >>> y_ref = np.array([0.01, 0.02, 0.01])
        >>> t_ref = np.array([0.0, 0.001, 0.002])
        >>> si = SystemIdentification(i_ref, y_ref, t_ref)
        >>> si.identify(method='de') # doctest: +SKIP
        """
        self._last_method = method.lower()
        self.function_call = 0

        if self._last_method == "de":
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
            G = Kp * (1 + s * Tz) / ((1 + s * Tp1) * (1 + s * Tp2))
            return G

        elif self._last_method == "slsqp":
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
            G = Kp * (1 + s * Tz) / ((1 + s * Tp1) * (1 + s * Tp2))
            return G

        elif self._last_method == "ss":
            p_values = kwargs.get("p", 20)
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

            costs = []
            p_range = p_values

            best_cost = np.inf
            best_p = None
            best_A = None
            best_B = None

            for p in p_range:
                if p >= N:
                    continue

                Y_X = np.zeros((N - p, 2))
                Phi_X = np.zeros((N - p, 2 + p))

                for k in range(N - p):
                    # Desired final state (k + p)
                    Y_X[k, :] = np.array([self.y_ref[k + p], self.v_ref[k + p]])

                    # Initial window state and the p subsequent inputs
                    x_k = [self.y_ref[k], self.v_ref[k]]
                    u_window = self.i_ref[k : k + p].tolist()
                    Phi_X[k, :] = np.array(x_k + u_window)

                # Solve secure Least Squares problem
                Theta_X, _, _, _ = np.linalg.lstsq(Phi_X, Y_X, rcond=None)
                Theta_X_T = np.transpose(Theta_X)

                # Extract A and B
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
                costs.append(cost)

                if cost < best_cost:
                    best_cost = cost
                    best_p = p
                    best_A = A
                    best_B = B

            if best_p is None:
                raise ValueError(
                    "All values in 'p' must be smaller than the data length."
                )

            # Updating the best values found
            self.p = best_p
            self.A = best_A
            self.B = best_B

            toc = time.time()

            print(f"State-Space identification finished in: {toc - tic:.2f} s")
            print(
                f"Identification finished. Best p = {self.p} with cost = {best_cost:.4f}"
            )

            dt = self.t_ref[1] - self.t_ref[0]
            C = np.array([[1, 0]])
            D = np.array([[0]])
            G_ss = ct.ss(self.A, self.B, C, D, dt=dt)
            return G_ss

        else:
            raise ValueError(f"Unknown method '{method}'. Use 'de', 'slsqp' or 'ss'.")
