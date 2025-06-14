import numpy as np
from scipy.optimize import root_scalar

class Kinetics():
    
    def __init__(
            self,
            t_half: float,            
            f: float,
            Vd: float,
            Tmax: float,
            Cmax: float,
            auc: float,
            optimize_start,
            t_half_pro,
            Tmax_pro
    ):
        self.t_half = t_half
        self.t_half_pro = t_half_pro
        self.f = f
        self.Vd = Vd
        self.Tmax = Tmax
        self.Tmax_pro = Tmax_pro
        self.Cmax = Cmax
        self.auc = auc

        def get_ka(ke, tmax):
            def tmax_equation(ka, ke, tmax):
                return (np.log(ka) - np.log(ke)) / (ka - ke) - tmax

            sol = root_scalar(tmax_equation, args=(ke, tmax), bracket=[ke + 0.01, 3], method='brentq')
            ka = sol.root if sol.converged else 1.0
            return ka

        self.ke = np.log(2) / t_half
        self.ke_pro = np.log(2) / t_half_pro
        self.ka = get_ka(self.ke, Tmax)
        self.optimize_start = optimize_start
    
    def set_michaelis_params(self, result_func, model_func):
        result = result_func(self, model_func)
        self.Vmax = result.x[0]
        self.Km = result.x[1]