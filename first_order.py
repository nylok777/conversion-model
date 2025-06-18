from collections.abc import Sequence
import numpy as np
from matplotlib import pyplot as plt
from optimizeModel import solve_odes, optimize_michaelis_menten_kinetics
from kinetics import Kinetics

def first_order_model(t, y, ka, ke, f, ):
    Agi, Ap = y

    d_Agi_dt = -ka*Agi
    d_Ap_dt = f*ka*Agi * np.e**(-ke)

    return (d_Agi_dt, d_Ap_dt)