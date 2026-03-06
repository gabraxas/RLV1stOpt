import numpy as np
from math import pi
from dataclasses import dataclass
from typing import Tuple

from scipy.integrate import solve_ivp
from scipy.optimize import minimize


# ---- 전역 파라미터 (MATLAB의 global A T Isp initial drymass xf 에 대응) ----
@dataclass
class GlobalParams:
    A: float = None
    T: float = None
    Isp: float = None
    initial: np.ndarray = None
    drymass: float = None
    xf: float = None


G = GlobalParams()


def optim_sl(vehicle: str):
    """
    MATLAB: function output = optim_sl(vehicle)
    를 Python으로 변환한 함수입니다.
    vehicle: "Falcon 9 ", "Electron ", "New Shepard" 중 하나.
    """
    # ---------------- 차량별 파라미터 설정 ----------------
    if vehicle == "Falcon 9 ":
        G.A = pi * 3.66 ** 2 / 4
        G.T = 5886000
        G.Isp = 282
        G.initial = np.array([36022, 60708, 1052, 1060, 76501], dtype=float)
        G.drymass = 25600
        G.xf = 200000
    elif vehicle == "Electron ":
        G.T = 224300
        G.Isp = 311
        G.A = pi * 1.2 ** 2 / 4
        G.initial = np.array([94824, 106485, 1914, 1463, 3950], dtype=float)
        G.drymass = 950
        G.xf = 480000
    elif vehicle == "New Shepard":
        G.T = 2201000
        G.Isp = 260
        G.A = pi * 7 ** 2 / 4
        G.initial = np.array([65000, 71000, 1500, 1322, 41000], dtype=float)
        G.drymass = 20569
        G.xf = 439000
    else:
        raise ValueError(f"Unknown vehicle: {vehicle}")

    # ----------------------- 최적화 설정 -----------------------
    lb = np.array([0.0, 100.0, 150.0])    # 하한
    ub = np.array([200.0, 400.0, 600.0])  # 상한
    x0 = np.array([20.0, 180.0, 310.0])   # 초기 추정

    # SciPy의 minimize를 사용해 fmincon과 비슷하게 작성
    # (제곱벌, 제약조건은 constraint 함수 사용)
    def objective(x):
        return objectivefunc_rls(x)

    # 부등식 제약 c(x) <= 0
    def constraint_fun(x):
        c, _ = constraint_rls(x)
        # SciPy의 'ineq' 타입은 fun(x) >= 0 형식을 기대하므로 부호를 바꿔준다.
        # 여기서 c(x) <= 0  ->  -c(x) >= 0
        return -np.array(c, dtype=float)

    constraints = [
        {'type': 'ineq', 'fun': constraint_fun}
    ]

    # bounds는 (low, high) 튜플 리스트로
    bounds = [(float(l), float(u)) for l, u in zip(lb, ub)]

    # tol, maxiter 등은 MATLAB 설정을 적당히 대응
    res = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={
            'ftol': 1e-12,            # 대략 OptimalityTolerance 비슷
            'maxiter': 10000,
            'disp': True
        }
    )

    xsolution = res.x

    # ---------------- ODE 설정 및 통합 ----------------
    # opts = odeset('RelTol',1e-4,'AbsTol',1e-6);
    rtol = 1e-4
    atol = 1e-6

    # boost-back phase : [0, xsolution(1)]
    t_span1 = (0.0, xsolution[0])
    sol1 = solve_ivp(
        dynamicsboostback,
        t_span1,
        G.initial,
        rtol=rtol,
        atol=atol,
        dense_output=True
    )

    # xinit 계산 (줄 42)
    state1_end = sol1.sol(xsolution[0])  # shape (5,) at t=xsolution[0]
    vx_end = state1_end[2]
    vy_end = state1_end[3]
    v_end = np.sqrt(vx_end ** 2 + vy_end ** 2)
    gamma_end = np.arccos(vx_end / v_end)
    xinit = np.array([
        state1_end[0],
        state1_end[1],
        v_end,
        gamma_end,
        state1_end[4]
    ])

    # coasting phase : [xsolution(1), xsolution(2)]
    t_span2 = (xsolution[0], xsolution[1])
    sol2 = solve_ivp(
        dynamicscoast,
        t_span2,
        xinit,
        rtol=rtol,
        atol=atol,
        dense_output=True
    )

    # 중간 상태 (줄 46)
    state2_mid = sol2.sol(xsolution[1])
    xmid = np.array([
        state2_mid[0],
        state2_mid[1],
        state2_mid[2],
        state2_mid[3],
        state2_mid[4]
    ])

    # landing phase : [xsolution(2), xsolution(3)]
    t_span3 = (xsolution[1], xsolution[2])
    sol3 = solve_ivp(
        dynamicslanding,
        t_span3,
        xmid,
        rtol=rtol,
        atol=atol,
        dense_output=True
    )

    # ---------------- 각 phase 시간/상태 샘플링 ----------------
    t1 = np.linspace(0.0, xsolution[0], 50)
    t2 = np.linspace(xsolution[0], xsolution[1], 50)
    t3 = np.linspace(xsolution[1], xsolution[2], 50)

    # sol1(t), sol2(t), sol3(t)에서 상태 평가
    s1 = sol1.sol(t1)  # shape (5, len(t1))
    s2 = sol2.sol(t2)
    s3 = sol3.sol(t3)

    x1 = s1[0, :]
    x2 = s2[0, :]
    x3 = s3[0, :]

    y1 = s1[1, :]
    y2 = s2[1, :]
    y3 = s3[1, :]

    # v1 = sqrt(vx^2 + vy^2)
    vx1 = s1[2, :]
    vy1 = s1[3, :]
    v1 = np.sqrt(vx1 ** 2 + vy1 ** 2)
    v2 = s2[2, :]
    v3 = s3[2, :]

    m1 = s1[4, :]
    m2 = s2[4, :]
    m3 = s3[4, :]

    gamma1 = np.arctan2(vy1, vx1)
    gamma2 = s2[3, :]
    gamma3 = s3[3, :]

    # 전체를 concatenate
    t = np.concatenate([t1, t2, t3])
    x_all = np.concatenate([x1, x2, x3])
    y_all = np.concatenate([y1, y2, y3])
    v_all = np.concatenate([v1, v2, v3])
    gamma_all = np.concatenate([gamma1, gamma2, gamma3])
    m_all = np.concatenate([m1, m2, m3])

    # MATLAB의 output = [t; x; y; v; gamma; m];
    # Python에서는 (6, N) array 혹은 dict로 반환
    output = np.vstack([t, x_all, y_all, v_all, gamma_all, m_all])
    return output, xsolution, res


# ---------------------- 제약조건 함수 ----------------------
def constraint_rls(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    MATLAB:
    function [c,ceq] = constraint_rls(x)
    """
    # ceq 는 비어 있음
    ceq = np.array([])

    # boost-back phase
    t_span1 = (0.0, x[0])
    sol1 = solve_ivp(
        dynamicsboostback,
        t_span1,
        G.initial,
        dense_output=True
    )

    s1_end = sol1.sol(x[0])
    vx_end = s1_end[2]
    vy_end = s1_end[3]
    vinit = np.sqrt(vx_end ** 2 + vy_end ** 2)
    g = np.arccos(vx_end / vinit)

    xinit = np.array([
        s1_end[0],
        s1_end[1],
        vinit,
        g,
        s1_end[4]
    ])

    # coasting
    t_span2 = (x[0], x[1])
    sol2 = solve_ivp(
        dynamicscoast,
        t_span2,
        xinit,
        dense_output=True
    )

    s2_mid = sol2.sol(x[1])
    xmid = np.array([
        s2_mid[0],
        s2_mid[1],
        s2_mid[2],
        s2_mid[3],
        s2_mid[4]
    ])

    # landing
    t_span3 = (x[1], x[2])
    sol3 = solve_ivp(
        dynamicslanding,
        t_span3,
        xmid,
        dense_output=True
    )

    # MATLAB: c(1)= -1*sol3.y(2,end);
    #         c(2)= drymass-sol3.y(5,end);
    y_end = sol3.y[:, -1]
    c1 = -1.0 * y_end[1]
    c2 = G.drymass - y_end[4]
    c = np.array([c1, c2])

    return c, ceq


# ---------------------- 목적함수 ----------------------
def objectivefunc_rls(x: np.ndarray) -> float:
    """
    MATLAB:
    function f = objectivefunc_rls(x)
    """
    # boost-back phase
    t_span1 = (0.0, x[0])
    sol1 = solve_ivp(
        dynamicsboostback,
        t_span1,
        G.initial,
        dense_output=True
    )

    s1_end = sol1.sol(x[0])
    vx_end = s1_end[2]
    vy_end = s1_end[3]
    v_end = np.sqrt(vx_end ** 2 + vy_end ** 2)
    gamma_end = np.arcsin(vy_end / v_end)

    xinit = np.array([
        s1_end[0],
        s1_end[1],
        v_end,
        gamma_end,
        s1_end[4]
    ])

    # coasting phase
    t_span2 = (x[0], x[1])
    sol2 = solve_ivp(
        dynamicscoast,
        t_span2,
        xinit,
        dense_output=True
    )

    s2_mid = sol2.sol(x[1])
    xmid = np.array([
        s2_mid[0],
        s2_mid[1],
        s2_mid[2],
        s2_mid[3],
        s2_mid[4]
    ])

    # landing phase
    t_span3 = (x[1], x[2])
    sol3 = solve_ivp(
        dynamicslanding,
        t_span3,
        xmid,
        dense_output=True
    )

    xland = sol3.y[:, -1]  # [x, y, v, gamma, m] at landing

    s1 = 400.0
    s2 = 100.0
    s3 = 800.0
    s4 = 100.0

    # f = -xland(5) + s1*(xf-xland(1))^2 + s2*(xland(2))^2  + ...
    #     s3*(xland(3)*sin(xland(4)))^2 + s4*(xland(3)*cos(xland(4)))^2
    x_pos = xland[0]
    y_pos = xland[1]
    v = xland[2]
    gamma = xland[3]
    m = xland[4]

    f = (
        -m
        + s1 * (G.xf - x_pos) ** 2
        + s2 * (y_pos) ** 2
        + s3 * (v * np.sin(gamma)) ** 2
        + s4 * (v * np.cos(gamma)) ** 2
    )
    return float(f)


# ---------------------- 동역학: coasting ----------------------
def dynamicscoast(t: float, x: np.ndarray) -> np.ndarray:
    """
    MATLAB:
    function f = dynamicscoast( t, x)
        state: [x, y, v, gamma, m]
    """
    g0 = 9.80665
    Cd = 0.75
    rho0 = 1.225
    h0 = 7500.0
    Re = 6378000.0

    f1 = x[2] * np.cos(x[3])
    f2 = x[2] * np.sin(x[3])
    f3 = 1.0 / x[4] * (-0.5 * rho0 * Cd * G.A * np.exp(-x[1] / h0) * (x[2] ** 2)) - g0 * np.sin(x[3])
    f4 = -1.0 / x[***

