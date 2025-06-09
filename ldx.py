import numpy as np
from optimizeModel import solve_odes, get_michaelis_menten_params
from matplotlib import pyplot as plt

def ldx_model(t, y, ka, ke, ke_ldx, f, Vmax, Km):
    ldx_gi, ldx_p, dex_p = y

    conv = (Vmax * ldx_p) / (Km + ldx_p)

    try:
        ldx_gi_dt = -ka * ldx_gi
        ldx_p_dt = ka * ldx_gi - conv - ke_ldx * ldx_p
        dex_p_dt = f* conv - (ke * dex_p)
    except Exception():
        return "LDX elimination cannot be None"
    
    return [ldx_gi_dt, ldx_p_dt, dex_p_dt]

tmax_target = 3.78
cmax_target = 69.3
auc_target = 1020

start = [35000, 30000]

ldx_dose_mg = 70
ldx_dose_ug = ldx_dose_mg * 1000

ke = np.log(2) / 10

V = 195_000
ke_ldx = 0.9
f = 0.297

result = get_michaelis_menten_params(
    start,
    ldx_model,
    (0, 72),
    (20_000, 70_000),
    (10_000, 60_000),
    V,
    ke,
    f,
    ldx_dose_ug,
    tmax_target,
    cmax_target,
    auc_target,
    ke_ldx
)

print("Vmax:", result.x[0])
print("Km:", result.x[1])


Vmax, Km = result.x

y0 = [70_000, 0, 0]

t_span = (0, 14)
t_eval = np.linspace(*t_span, 2000)

solution = solve_odes(ldx_model, t_span, y0, 0.6691, ke, ke_ldx, f, Vmax, Km)

t = solution.t
y = solution.y

Cdex_ng = (y[2] / V) * 1000


plt.plot(t, Cdex_ng)
plt.axhline(cmax_target, color='gray', linestyle='--', label='Cmax target')
plt.axvline(tmax_target, color='red', linestyle='--', label='Tmax target')
plt.xlabel("Time (hours)")
plt.ylabel("d-Amphetamine (ng/mL)")
plt.title(f"Plasma d-Amphetamine after {y0[0]/1000} mg LDX")
plt.grid(True)
plt.show()