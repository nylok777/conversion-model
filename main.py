from kinetics import KineticsMM
from ldx import ldx_model, get_result_ldx, simulate, plot_last_dose

kinetics = KineticsMM(
    0.297,
    0.9,
    1.15,
    11.3,
    195_000,
    4,
    44.6,
    763.1
)

kinetics.get_michaelis_params(get_result_ldx, ldx_model, (0, 96), 50_000, (25_000, 20_000))

full = [22, 28.84, 24.083, 23.684, 23.54, 24.116, 21.384, 24.6, 28.216, 22.25]

results = simulate(ldx_model, kinetics, full[-1], 30, full[:-2])
y = results[0]
y0 = [30_000, y[-2][-1], y[-1][-1]]
plot_last_dose(y0, 30, 10, kinetics)