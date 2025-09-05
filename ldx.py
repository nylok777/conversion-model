from collections.abc import Sequence, Callable
from collections import deque
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult, minimize
from kinetics import KineticsFromProDrug


def ldx_model(t, y, ka, ke, Vmax, Km, efficiency, ke_ldx):
    ldx_gi, ldx_p, dex_p = y

    conv = (Vmax * ldx_p) / (Km + ldx_p)

    ldx_gi_dt = -ka * ldx_gi
    ldx_p_dt = ka * ldx_gi - conv - ke_ldx * ldx_p
    dex_p_dt = efficiency * conv - (ke * dex_p)

    return (ldx_gi_dt, ldx_p_dt, dex_p_dt)


class Ldx(KineticsFromProDrug):
    def __init__(self, ldx_model: Callable, **kwargs):
        super().__init__(**kwargs)
        self.ldx_model = ldx_model

    def _solve_odes(
        self,
        t_span: Sequence[float],
        y0: Sequence[float],
    ):
        t_eval = np.linspace(*t_span, 10_000)  # type: ignore

        solution = solve_ivp(
            fun=self.ldx_model,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            args=(self.ka, self.ke, self.Vmax, self.Km, self.efficiency, self.ke_pro),
        )

        return solution

    def _cost_function_mm(
        self,
        params: tuple[float, float],
        t_span: Sequence[float],
        dose_ug: float,
        wt: float = 10.0,
        wc: float = 1.0,
        wa: float = 1.0,
        eps=1e-6,
    ) -> float:
        Vmax, Km = params
        self.Vmax, self.Km = Vmax, Km
        y0 = (dose_ug, 0, 0)

        solution = self._solve_odes(t_span, y0)
        t = solution.t

        C_active_ng = (solution.y[2] / self.Vd) * 1000

        idx = np.argmax(C_active_ng)
        tmax = t[idx]
        cmax = C_active_ng[idx]
        auc = np.trapezoid(C_active_ng, t)

        err_t = (np.log(tmax + eps) - np.log(self.Tmax + eps)) ** 2
        err_c = (np.log(cmax + eps) - np.log(self.Cmax + eps)) ** 2
        err_a = (np.log(auc + eps) - np.log(self.auc + eps)) ** 2

        return wt * err_t + wc * err_c + wa * err_a

    def _optimize_michaelis_menten_kinetics(
        self,
        initial_guess: Sequence[float],
        t_span: Sequence[float],
        dose_ug: float,
    ) -> OptimizeResult:
        result = minimize(
            fun=self._cost_function_mm,
            x0=initial_guess,
            args=(t_span, dose_ug),
            method="Nelder-Mead",
        )

        return result

    def find_mm_params(
        self,
        t_span: tuple[float, float] | tuple[int, int],
        dose_ug: float | int,
        initial_guess: Sequence[float] | Sequence[int],
    ):
        result = self._optimize_michaelis_menten_kinetics(
            initial_guess=initial_guess,
            t_span=t_span,
            dose_ug=dose_ug,
        )
        return result

    def _integrate_time_plasma_conc(
        self,
        t_start: float,
        t_end: float,
        dose_mg: float | int,
        y0: Sequence[float] | None = None,
    ) -> tuple:
        if y0 is None:
            y0 = [dose_mg * 1000, 0, 0]

        t_span = (t_start, t_end)

        solution = self._solve_odes(
            t_span,
            y0,
        )

        t = solution.t
        y = solution.y

        Cdex_ng = (y[2] / self.Vd) * 1000

        return (y, t, Cdex_ng)

    def simulate(
        self,
        t_end: float,
        doses_mg: tuple,
        times_btwn_doses: None | list,
    ):
        if times_btwn_doses is None:
            return self._integrate_time_plasma_conc(0, t_end, doses_mg[0])
        elif type(times_btwn_doses) is list:
            return self._simulate_multiple_doses(t_end, doses_mg, times_btwn_doses)

    def _simulate_multiple_doses(
        self,
        t_end: float,
        doses: tuple,
        times_btwn_doses: list,
    ):
        times_btwn_doses.append(t_end)
        doses_que = deque(doses)
        t_next = 0
        y = [[0], [0], [0]]
        t_all = ng_all = []
        y_all = [[0]]

        if len(doses_que) < 2:
            dose_mg = doses_que.popleft()
            dose_ug = dose_mg * 1000
        else:
            dose_mg = dose_ug = 0

        for time in times_btwn_doses:
            try:
                dose_mg = doses_que.popleft()
                dose_ug = dose_mg * 1000
            except IndexError:
                pass

            y, t, ng = self._integrate_time_plasma_conc(
                t_start=t_next,
                t_end=t_next + time,
                dose_mg=dose_mg,
                y0=[y[0][-1] + dose_ug, y[1][-1], y[2][-1]],
            )

            t_next = t_next + time
            y_all = np.concatenate([y_all, y], axis=None)
            t_all = np.concatenate([t_all, t], axis=None)
            ng_all = np.concatenate([ng_all, ng], axis=None)

        return (y_all, t_all, ng_all)
