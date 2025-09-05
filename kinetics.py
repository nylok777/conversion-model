from collections.abc import Sequence
from collections.abc import Callable
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.optimize import OptimizeResult, root_scalar


class Kinetics(metaclass=ABCMeta):
    def __init__(
        self,
        t_half: float,
        Vd: float,
        Tmax: float,
        Cmax: float,
        auc: float,
    ):
        self.t_half = t_half
        self.Vd = Vd
        self.Tmax = Tmax
        self.Cmax = Cmax
        self.auc = auc
        self.ke = np.log(2) / t_half

        def get_ka(ke, tmax):
            def tmax_equation(ka, ke, tmax):
                return (np.log(ka) - np.log(ke)) / (ka - ke) - tmax

            sol = root_scalar(
                tmax_equation, args=(ke, tmax), bracket=[ke + 0.01, 3], method="brentq"
            )
            ka = sol.root if sol.converged else 1.0
            return ka

        self.ka = get_ka(self.ke, Tmax)


class KineticsFO(Kinetics):
    def __init__(self, t_half, Vd, Tmax, Cmax, auc, bioaval):
        super().__init__(t_half, Vd, Tmax, Cmax, auc)
        self.bioaval = bioaval


class KineticsMM(Kinetics):
    def __init__(self, t_half, Vd, Tmax, Cmax, auc):
        super().__init__(t_half, Vd, Tmax, Cmax, auc)

    def read_mm_params_from_file(self, file_path: str):
        try:
            results = np.load("params.npy")

            self.Vmax = results[0]
            self.Km = results[1]
        except FileNotFoundError:
            print("file not found")

    @abstractmethod
    def _solve_odes(
        self,
        t_span: Sequence[float],
        y0: Sequence[float],
    ): ...

    @abstractmethod
    def _cost_function_mm(
        self,
        params: tuple[float, float],
        t_span: Sequence[float],
        dose_ug: float,
        wt: float = 10.0,
        wc: float = 1.0,
        wa: float = 1.0,
        eps=1e-6,
    ) -> float | int | Sequence: ...

    @abstractmethod
    def _optimize_michaelis_menten_kinetics(
        self,
        initial_guess: Sequence[float],
        t_span: Sequence[float],
        dose_ug: float,
    ) -> OptimizeResult: ...


class KineticsFromProDrug(KineticsMM):
    def __init__(
        self,
        efficiency: float,
        t_half_pro: float,
        Tmax_pro: float,
        t_half: float,
        Vd: float,
        Tmax: float,
        Cmax: float,
        auc: float,
    ):
        super().__init__(t_half, Vd, Tmax, Cmax, auc)

        self.t_half_pro = t_half_pro
        self.efficiency = efficiency
        self.Tmax_pro = Tmax_pro
        self.ke_pro = np.log(2) / t_half_pro

    def read_mm_params_from_file(self, file_path: str):
        return super().read_mm_params_from_file(file_path)
