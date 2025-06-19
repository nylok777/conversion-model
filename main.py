from kinetics import Kinetics
from ldx import ldx_model, get_result_ldx, simulate

kinetics = Kinetics(
    11.3,
    0.297,
    195_000,
    4,
    44.6,
    763.1,
    0.9,
    1.15
)

kinetics.get_michaelis_params(get_result_ldx, ldx_model, (0, 96), 50_000, (25_000, 20_000))

full = [22, 28.84, 24.083, 23.684, 23.54, 24.116, 21.384, 24.6]

simulate(ldx_model, kinetics, 14, 30, full, 24)