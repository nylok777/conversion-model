from matplotlib import pyplot as plt
import numpy as np


def plot_time_plasma_conc(
    t,
    y,
    user_dose: float,
    substance: dict,
    xticks_tspan: float | None,
):
    plt.plot(t, y)
    plt.xlabel("Time (hours)")
    plt.ylabel(f"{substance['active']} (ng/mL)")
    if xticks_tspan is not None:
        plt.xticks(np.arange(xticks_tspan + 1))

    plt.ylabel(f"{substance['active']} (ng/mL)")
    if "prodrug" in substance:
        plt.title(
            f"Plasma {substance['active']} after {user_dose} mg of {substance['prodrug']}"
        )
    else:
        plt.title(f"Plasma {substance['active']} after {user_dose} mg")
    plt.grid(True)
    plt.show()


def test_plot(t, Cdex_ng, dose: float, tmax_target: float, cmax_target: float):
    plt.plot(t, Cdex_ng)
    plt.xlabel("Time (hours)")
    plt.ylabel("(ng/mL)")
    plt.axhline(cmax_target, color="gray", linestyle="--", label="Cmax target")
    plt.axvline(tmax_target, color="red", linestyle="--", label="Tmax target")
    plt.title(f"Plasma concentration after {dose} mg")
    plt.grid(True)
    plt.show()
