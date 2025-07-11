from collections.abc import Sequence
from collections.abc import Callable
from abc import ABCMeta
import numpy as np
from scipy.optimize import root_scalar


class Kinetics(metaclass=ABCMeta):
    def __init__(self, t_half: float, Vd: float, Tmax: float, bioaval: float, Cmax: float=None, auc: float=None):
        self.t_half = t_half
        self.Vd = Vd
        self.Tmax = Tmax
        self.Cmax = Cmax
        self.auc = auc
        self.bioaval = bioaval
        self.ke = np.log(2) / t_half

        def get_ka(ke, tmax):
            def tmax_equation(ka, ke, tmax):
                return (np.log(ka) - np.log(ke)) / (ka - ke) - tmax

            sol = root_scalar(tmax_equation, args=(ke, tmax), bracket=[ke + 0.01, 3], method='brentq')
            ka = sol.root if sol.converged else 1.0
            return ka
        
        self.ka = get_ka(self.ke, Tmax)

class KineticsFO(Kinetics):
    def __init__(self, t_half, Vd, Tmax, bioaval, Cmax = None, auc = None):
        super().__init__(t_half, Vd, Tmax, bioaval, Cmax, auc)

class KineticsMM(Kinetics):
    def __init__(self, t_half, Vd, Tmax, bioaval, Cmax = None, auc = None):
        super().__init__(t_half, Vd, Tmax, bioaval, Cmax, auc)

    def get_michaelis_params(self, result_func: callable, model_func: callable, t_span: Sequence[float, float], dose_ug: float,
                             optimize_start: Sequence[float, float]):
        try:
            f = open('params.txt')
            results = f.read()
            f.close()            
            results = results.split(',')            
            self.Vmax = float(results[0])
            self.Km = float(results[1])
        except FileNotFoundError:
            result = result_func(self, model_func, t_span, dose_ug, optimize_start)
            self.Vmax = result.x[0]
            self.Km = result.x[1]
            results = (result.x[0], result.x[1])
            with open('params.txt', mode='x') as f:
                f.write(str(results).strip('()'))

class KineticsFromProDrug(KineticsMM):
    def __init__(
            self,
            efficiency: float,
            t_half_pro: float,
            Tmax_pro: float,
            t_half: float,
            Vd: float,
            Tmax: float,
            bioaval: float,
            Cmax: float,
            auc: float
        ):
        super().__init__(t_half, Vd, Tmax, bioaval, Cmax, auc)
        
        self.t_half_pro = t_half_pro
        self.efficiency = efficiency
        self.Tmax_pro = Tmax_pro
        self.ke_pro = np.log(2) / t_half_pro
        
    def get_michaelis_params(self, result_func, model_func, t_span, dose_ug, optimize_start):
        return super().get_michaelis_params(result_func, model_func, t_span, dose_ug, optimize_start)