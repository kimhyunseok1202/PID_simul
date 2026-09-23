import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(layout="wide", page_title="FOPDT 공정 PID 튜닝 시뮬레이터")
st.title("🎛️ FOPDT 공정 & 실시간 PID 튜닝 시뮬레이터")

# ---------------------------------------------------------
# 좌측 사이드바: 공정 및 PID 튜닝 파라미터 조작 슬라이더
# ---------------------------------------------------------
st.sidebar.header("1. 공정 파라미터 설정")
setpoint = st.sidebar.slider("목표 온도 (℃)", 100.0, 1500.0, 1000.0, 10.0)
tau = st.sidebar.slider("시상수 τ (s)", 50.0, 600.0, 300.0, 10.0)
K_gain = st.sidebar.slider("공정 게인 K", 1.0, 30.0, 16.0, 1.0)
delay_time = st.sidebar.slider("시간 지연 θ (s)", 0.0, 100.0, 30.0, 5.0)
T_ambient = st.sidebar.slider("초기/외기 온도 (℃)", 0.0, 50.0, 25.0, 1.0)

st.sidebar.header("2. PID 파라미터 튜닝")
Kc = st.sidebar.slider("비례 게인 Kc", 0.01, 2.0, 0.3, 0.02)
tau_I = st.sidebar.slider("적분 시간 τI (s)", 10.0, 1000.0, 200.0, 10.0)
tau_D = st.sidebar.slider("미분 시간 τD (s)", 0.0, 100.0, 25.0, 1.0)

# ---------------------------------------------------------
# 시뮬레이션 계산 (오일러법)
# ---------------------------------------------------------
dt = 1.0
total_time = 5400
n_steps = int(total_time / dt)
time = np.linspace(0, total_time, n_steps + 1)
delay_steps = int(delay_time / dt)

T = np.zeros(n_steps + 1)
u = np.zeros(n_steps + 1)
T[0] = T_ambient

integral = 0.0
prev_error = 0.0

for k in range(n_steps):
    e = setpoint - T[k]

    # PID 계산 및 안티와인드업
    P_term = Kc * e
    D_term = Kc * tau_D * ((e - prev_error) / dt if k > 0 else 0.0)

    I_term = (Kc / tau_I) * integral if tau_I > 0 else 0.0
    u_unsat = P_term + I_term + D_term

    if not ((u_unsat >= 100.0 and e > 0) or (u_unsat <= 0.0 and e < 0)):
        integral += e * dt

    I_term = (Kc / tau_I) * integral if tau_I > 0 else 0.0
    u_curr = np.clip(P_term + I_term + D_term, 0.0, 100.0)
    prev_error = e
    u[k] = u_curr

    # FOPDT 모델 지연시간 적용 갱신
    u_delayed = u[k - delay_steps] if k >= delay_steps else 0.0
    dT_dt = (-(T[k] - T_ambient) + K_gain * u_delayed) / tau
    T[k + 1] = T[k] + dt * dT_dt

u[-1] = u[-2]

# 성능 지표
final_error = setpoint - T[-1]
overshoot = max(0.0, np.max(T) - setpoint) if np.max(T) > setpoint else 0.0

# ---------------------------------------------------------
# 결과 화면 및 실시간 그래프 출력
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("최종 도달 온도", f"{T[-1]:.2f} ℃")
col2.metric("최종 오차 (Final Error)", f"{final_error:.2f} ℃")
col3.metric("최대 오버슈트 (Overshoot)", f"{overshoot:.2f} ℃")

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
)

ax1.plot(time, T, "m-", linewidth=2, label="PID 제어 온도")
ax1.axhline(
    setpoint,
    color="r",
    linestyle="--",
    linewidth=1.2,
    label=f"목표치 ({setpoint}℃)",
)
ax1.set_ylabel("온도 (℃)")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="lower right")

ax2.plot(time, u, "g-", linewidth=1.5, label="히터 출력 u (%)")
ax2.set_xlabel("시간 (s)")
ax2.set_ylabel("출력 (%)")
ax2.set_ylim([-5, 105])
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper right")

st.pyplot(fig)
