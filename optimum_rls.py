import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# 전역 변수에 해당하는 값들을 모아둘 딕셔너리
globals_state = {
    "A": None,
    "T": None,
    "Isp": None,
    "initial": None,
    "drymass": None,
}

def set_vehicle(vehicle: str):
    """MATLAB의 vehicle if-elseif 블록에 해당."""
    g = globals_state
    if vehicle == "Falcon 9 ":
        g["A"] = np.pi * 3.66**2 / 4
        g["T"] = 5886000.0
        g["Isp"] = 282.0
        g["initial"] = np.array([36022, 60708, 1052, 1060, 76501], dtype=float)
        g["drymass"] = 25600.0
    elif vehicle == "Electron ":
        g["T"] = 224300.0
        g["Isp"] = 311.0
        g["A"] = np.pi * 1.2**2 / 4
        g["initial"] = np.array([94824, 106485, 1914, 1463, 3950], dtype=float)
        g["drymass"] = 950.0
    elif vehicle == "New Shepard":
        g["T"] = 2201000.0
        g["Isp"] = 260.0
        g["A"] = np.pi * 7.0**2 / 4
        g["initial"] = np.array([65000, 71000, 1500, 1322, 41000], dtype=float)
        g["drymass"] = 20569.0
    else:
        raise ValueError(f"Unknown vehicle: {vehicle}")


def dynamics_coast(t, x):
    # x = [x, y, v, gamma, m]
    g0 = 9.80665
    Cd = 0.75
    rho = 1.225
    h0 = 7500.0
    Re = 6378000.0
    A = globals_state["A"]

    f1 = x[2] * np.cos(x[3])
    f2 = x[2] * np.sin(x[3])
    f3 = (1.0 / x[4]) * (-0.5 * rho * Cd * A * np.exp(-x[1] / h0) * x[2] ** 2) - g0 * np.sin(x[3])
    f4 = -(1.0 / x[2]) * (g0 - x[2] ** 2 / (Re + x[1])) * np.cos(x[3])
    f5 = 0.0
    return [f1, f2, f3, f4, f5]


def dynamics_boostback(t, x):
    # x = [x, y, vx, vy, m]
    g0 = 9.80665
    Cd = 0.75
    rho = 1.225
    h0 = 7500.0
    Re = 6378000.0
    K = 1.0
    theta = np.pi  # during boostback only x motion is restricted
    A = globals_state["A"]
    T = globals_state["T"]
    Isp = globals_state["Isp"]

    v_mag = np.sqrt(x[2] ** 2 + x[3] ** 2) + 1e-12  # 0 division 방지용 작은 값

    f1 = x[2]
    f2 = x[3]
    f3 = (1.0 / x[4]) * (
        K * T / 3.0 * np.cos(theta)
        - 0.5 * rho * np.exp(-x[1] / h0) * Cd * A * (x[2] ** 2 + x[3] ** 2) * x[2] / v_mag
    )
    f4 = (1.0 / x[4]) * (
        K * T / 3.0 * np.sin(theta)
        - 0.5 * rho * Cd * A * np.exp(-x[1] / h0) * (x[2] ** 2 + x[3] ** 2) * x[3] / v_mag
    ) - g0
    f5 = -T***

