from kinetics import Kinetics
from ldx import ldx_model, get_result_ldx, show_plot_to_user

kinetics = Kinetics(
    10,
    0.297,
    195_000,
    3.78,
    69.3,
    1020,
    (35_000, 30_000),
    0.9,
    1.15
)

kinetics.set_michaelis_params(get_result_ldx, ldx_model)

show_plot_to_user(ldx_model, kinetics)