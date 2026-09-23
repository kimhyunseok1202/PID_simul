import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(layout="wide", page_title="실시간 PID 튜닝 시뮬레이터")
st.title("🎛️ 실시간 PID 튜닝 시뮬레이터")

st.sidebar.header("1. 공정 파라미터 설정")

setpoint = st.sidebar.number_input("목표 온도 (℃)", min_value=100.0, max_value=1500.0, value=1000.0, step=10.0)
tau = st.sidebar.number_input("시상수 τ (s)", min_value=10.0, max_value=1000.0, value=300.0, step=10.0)
K_gain = st.sidebar.number_input("공정 게인 K", min_value=1.0, max_value=50.0, value=16.0, step=1.0)
delay_time = st.sidebar.number_input("시간 지연 θ (s)", min_value=0.0, max_value=200.0, value=30.0, step=1.0)
T_ambient = st.sidebar.number_input("초기/외기 온도 (℃)", min_value=0.0, max_value=100.0, value=25.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.header("2. PID 파라미터 튜닝")

Kc = st.sidebar.number_input("비례 게인 Kc", min_value=0.001, max_value=5.0, value=0.3, step=0.01, format="%.3f")
tau_I = st.sidebar.number_input("적분 시간 τI (s)", min_value=1.0, max_value=2000.0, value=200.0, step=5.0)
tau_D = st.sidebar.number_input("미분 시간 τD (s)", min_value=0.0, max_value=200.0, value=25.0, step=1.0)

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

    u_delayed = u[k - delay_steps] if k >= delay_steps else 0.0
    dT_dt = (-(T[k] - T_ambient) + K_gain * u_delayed) / tau
    T[k + 1] = T[k] + dt * dT_dt

u[-1] = u[-2]

final_error = setpoint - T[-1]
overshoot = max(0.0, np.max(T) - setpoint) if np.max(T) > setpoint else 0.0

col1, col2, col3 = st.columns(3)
col1.metric("최종 도달 온도", f"{T[-1]:.2f} ℃")
col2.metric("최종 오차 (Final Error)", f"{final_error:.2f} ℃")
col3.metric("최대 오버슈트 (Overshoot)", f"{overshoot:.2f} ℃")

# 한 화면에 모두 들어오도록 높이를 4.2인치로 축소하고 여백 최적화
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

ax1.plot(time, T, 'm-', linewidth=2, label=f'PID 제어 온도 (최종값: {T[-1]:.2f}℃)')
ax1.axhline(setpoint, color='r', linestyle='--', linewidth=1.2, label=f'목표치 ({setpoint}℃)')
ax1.set_ylabel("온도 (℃)", fontsize=9)
ax1.tick_params(axis='both', labelsize=8)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc="lower right", fontsize=8)

ax2.plot(time, u, 'g-', linewidth=1.5, label='히터 출력 u (%)')
ax2.set_xlabel("시간 (s)", fontsize=9)
ax2.set_ylabel("출력 (%)", fontsize=9)
ax2.set_ylim([-5, 105])
ax2.tick_params(axis='both', labelsize=8)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc="upper right", fontsize=8)

plt.tight_layout()
st.pyplot(fig)
