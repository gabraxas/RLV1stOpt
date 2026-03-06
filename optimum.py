import numpy as np
from math import pi
from dataclasses import dataclass
from typing import Tuple

from scipy.integrate import solve_ivp
from scipy.optimize import minimize


# ---- MATLAB global T A Isp initial drymass 대응 ----
@dataclass
class OptGlobals:
    T: float = None
    A: float = None
    Isp: float = None
    initial: np.ndarray = None
    drymass: float = None


OG = OptGlobals()


def optimum(vehicle: str):
    """
    MATLAB: function output = optimum(vehicle)
    vehicle: "Falcon 9 ", "Electron ", "New Shepard"
    """

    # ---- 차량별 초기 조건 설정 (라인 8~25) ----
    if vehicle == "Falcon 9 ":
        OG.A = pi * 3.66 ** 2 / 4
        OG.T = 5886000
        OG.Isp = 282
        OG.initial = np.array([
            36022.0,
            60708.0,
            np.sqrt(1052.0 ** 2 + 1060.0 ** 2),
            np.arctan(1060.0 / 1052.0),
            76501.0
        ])
        OG.drymass = 25600.0

    elif vehicle == "Electron ":
        OG.T = 224300.0
        OG.Isp = 311.0
        OG.A = pi * 1.2 ** 2 / 4
        OG.initial = np.array([
            94824.0,
            106485.0,
            2409.0,
            37.4 * pi / 180.0,
            3950.0
        ])
        OG.drymass = 950.0

    elif vehicle == "New Shepard":
        OG.T = 2201000.0
        OG.Isp = 260.0
        OG.A = pi * 7.0 ** 2 / 4
        OG.initial = np.array([
            60000.0,
            70000.0,
            2000.0,
            np.arctan(529.0 / 600.0),
            30500.0
        ])
        OG.drymass = 20569.0

    else:
        raise ValueError(f"Unknown vehicle: {vehicle}")

    # ---- 최적화 설정 (라인 29~37) ----
    lb = np.array([0.0, 0.0])
    ub = np.array([400.0, 600.0])
    x0 = np.array([120.0, 300.0])  # 초기 추정

    def objective(x):
        return Objectivefunc(x)

    # MATLAB: c(x) <= 0  형식이므로 SciPy 'ineq' (fun(x) >= 0) 를 위해 부호 반전
    def ineq_constraint_fun(x):
        c, _ = constraints(x)
        return -np.array(c, dtype=float)

    constraints_sqp = [
        {'type': 'ineq', 'fun': ineq_constraint_fun}
    ]
    bounds = [(float(l), float(u)) for l, u in zip(lb, ub)]

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints_sqp,
        options={
            "ftol": 1e-12,
            "maxiter": 10000,
            "disp": True,
        },
    )

    xsolution = res.x

    # ---- 최적 해에 대해 ODE 적분 (라인 40~44) ----
    # coasting phase: [0, xsolution(1)]
    t_span1 = (0.0, xsolution[0])
    sol1 = solve_ivp(
        dynamicscoast,
        t_span1,
        OG.initial,
        dense_output=True
    )

    state_end1 = sol1.sol(xsolution[0])
    xinit = np.array([
        state_end1[0],
        state_end1[1],
        state_end1[2],
        state_end1[3],
        state_end1[4],
    ])

    # landing phase: [xsolution(1), xsolution(2)]
    t_span2 = (xsolution[0], xsolution[1])
    sol2 = solve_ivp(
        dynamicslanding,
        t_span2,
        xinit,
        dense_output=True
    )

    # ---- 결과 샘플링 및 output 구성 (라인 46~61) ----
    # 샘플 포인트 개수는 적당히 50개씩
    t1 = np.linspace(0.0, xsolution[0], 50)
    t2 = np.linspace(xsolution[0], xsolution[1], 50)

    s1 = sol1.sol(t1)  # (5, len(t1))
    s2 = sol2.sol(t2)  # (5, len(t2))

    x1 = s1[0, :]
    x2 = s2[0, :]

    y1 = s1[1, :]
    y2 = s2[1, :]

    v1 = s1[2, :]
    v2 = s2[2, :]

    gamma1 = s1[3, :]
    gamma2 = s2[3, :]

    m1 = s1[4, :]
    m2 = s2[4, :]

    t_all = np.concatenate([t1, t2])
    x_all = np.concatenate([x1, x2])
    y_all = np.concatenate([y1, y2])
    v_all = np.concatenate([v1, v2])
    gamma_all = np.concatenate([gamma1, gamma2])
    m_all = np.concatenate([m1, m2])

    # MATLAB: output = [t; x; y; v; gamma; m]
    output = np.vstack([t_all, x_all, y_all, v_all, gamma_all, m_all])

    return output, xsolution, res


# ---- dynamicslanding (라인 87~105) ----
def dynamicslanding(t: float, x: np.ndarray) -> np.ndarray:
    """
    상태: [x, y, v, gamma, m]
    """
    g0 = 9.80665
    Cd = 0.75
    rho = 1.225
    h0 = 7500.0
    Re = 6378000.0

    Beta = 0.5 * rho * Cd * OG.A

    f1 = x[2] * np.cos(x[3])
    f2 = x[2] * np.sin(x[3])
    f3 = 1.0 / x[4] * (-1.0 * OG.T / 9.0 - Beta * np.exp(-x[1] / h0) * (x[2] ** 2)) - g0 * np.sin(x[3])
    f4 = -1.0 / x[2] * (g0 - (x[2] ** 2) / (Re + x[1])) * np.cos(x[3])
    f5 = -1.0 * OG.T / 9.0 / (OG.Isp * g0)

    return np.array([f1, f2, f3, f4, f5])


# ---- dynamicscoast (라인 107~123) ----
def dynamicscoast(t: float, x: np.ndarray) -> np.ndarray:
    """
    상태: [x, y, v, gamma, m]
    """
    g0 = 9.80665
    Cd = 0.75
    rho = 1.225
    h0 = 7500.0
    Re = 6378000.0

    Beta = 0.5 * rho * Cd * OG.A

    f1 = x[2] * np.cos(x[3])
    f2 = x[2] * np.sin(x[3])
    f3 = 1.0 / x[4] * (-Beta * np.exp(-x[1] / h0) * (x[2] ** 2)) - g0 * np.sin(x[3])
    f4 = -1.0 / x[2] * (g0 - (x[2] ** 2) / (Re + x[1])) * np.cos(x[3])
    f5 = 0.0

    return np.array([f1, f2, f3, f4, f5])


# ---- constraints (라인 126~140) ----
def constraints(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    MATLAB:
    function [c,ceq] = constraints(x)
    """
    ceq = np.array([])

    # coasting phase [0, x(1)]
    t_span1 = (0.0, x[0])
    sol1 = solve_ivp(
        dynamicscoast,
        t_span1,
        OG.initial,
        dense_output=True
    )

    s1_end = sol1.sol(x[0])
    xinit = np.array([
        s1_end[0],
        s1_end[1],
        s1_end[2],
        s1_end[3],
        s1_end[4],
    ])

    # landing phase [x(1), x(2)]
    t_span2 = (x[0], x[1])
    sol2 = solve_ivp(
        dynamicslanding,
        t_span2,
        xinit,
        dense_output=True
    )

    y_end = sol2.y[:, -1]  # [x, y, v, gamma, m] at final

    # c(1)= -1*sol2.y(2,end);
    # c(2)= drymass-sol2.y(5,end);
    c1 = -1.0 * y_end[1]
    c2 = OG.drymass - y_end[4]
    c = np.array([c1, c2])

    return c, ceq


# ---- Objectivefunc (라인 142~157) ----
def Objectivefunc(x: np.ndarray) -> float:
    """
    MATLAB: function f = Objectivefunc(x)
    """
    # coasting phase
    t_span1 = (0.0, x[0])
    sol1 = solve_ivp(
        dynamicscoast,
        t_span1,
        OG.initial,
        dense_output=True
    )

    s1_end = sol1.sol(x[0])
    xinit = np.array([
        s1_end[0],
        s1_end[1],
        s1_end[2],
        s1_end[3],
        s1_end[4],
    ])

    # landing phase
    t_span2 = (x[0], x[1])
    sol2 = solve_ivp(
        dynamicslanding,
        t_span2,
        xinit,
        dense_output=True
    )

    xf = sol2.y[:, -1]  # [x, y, v, gamma, m] at final

    s1 = 10.5
    s2 = 1000.5
    s3 = 1000.5

    f = (
        -xf[4]
        + s1 * (xf[1] ** 2)
        + s2 * (xf[2] * np.sin(xf[3])) ** 2
        + s3 * (xf[2] * np.cos(xf[3])) ** 2
    )
    return float(f)
