import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# 리눅스 서버에 설치된 나눔글꼴 적용 (packages.txt의 fonts-nanum 기준)
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

# 브라우저 탭 및 메인 타이틀 설정
st.set_page_config(layout="wide", page_title="실시간 PID 튜닝 시뮬레이터")
st.title("🎛️ 실시간 PID 튜닝 시뮬레이터")

# ---------------------------------------------------------
# 상단: 시뮬레이터 사용 가이드라인 (접이식 안내창)
# ---------------------------------------------------------
with st.expander("📖 시뮬레이터 사용 가이드 및 PID 튜닝 요령 (클릭하여 열기)"):
    st.markdown("""
    ### 1. 시뮬레이터 조작 방법
    * **파라미터 입력**: 좌측 사이드바에서 수치를 직접 입력하거나 `+ / -` 버튼으로 변경합니다.
    * **그래프 표시 온/오프**: 좌측 사이드바의 '3. 그래프 표시 설정'에서 원하는 그래프만 선택적으로 켜고 끌 수 있습니다.
    * **실시간 갱신**: 파라미터나 체크박스를 변경하면 우측 화면이 즉시 업데이트됩니다.
### 2. 파라미터 용어 정리
* **공정 파라미터 (Process Parameters)**
    * **목표 온도 (Setpoint)**: 도달하고자 하는 목표 온도 (기본 1000℃)
    * **시상수 τ (Tau)**: 공정 열용량에 따른 반응 지연 (값이 클수록 반응이 느림)
    * **공정 게인 K (Gain)**: 히터 출력 1%당 상승할 수 있는 최대 온도
    * **시간 지연 θ (Delay)**: 조작 신호 후 실제 반응 시작까지 걸리는 불감 시간(Dead time)
* **PID 제어기 튜닝 파라미터 (Tuning Parameters)**
    * **비례 게인 \(K_c\) (Proportional)**: 현재 오차에 즉각 반응하는 세기. 너무 크면 격렬한 진동이 발생합니다.
    * **적분 시간 \(\tau_I\) (Integral)**: 잔류 오차를 누적 계산해 제거. 작을수록 적분 작용이 강해지며 목표치에 빠르게 수렴하지만 오버슈트가 발생할 수 있습니다.
    * **미분 시간 \(\tau_D\) (Derivative)**: 급격한 온도 변화율을 억제하는 브레이크 역할.
""")
---------------------------------------------------------
좌측 사이드바: 슬라이더 + 숫자 직접 입력 동시 지원
---------------------------------------------------------
st.sidebar.header("1. 공정 파라미터 설정")

목표 온도
setpoint = st.sidebar.number_input(
"목표 온도 (℃)",
min_value=100.0,
max_value=1500.0,
value=1000.0,
step=10.0,
)

시상수 tau
tau = st.sidebar.number_input(
"시상수 τ (s)", min_value=10.0, max_value=1000.0, value=300.0, step=10.0
)

공정 게인 K
K_gain = st.sidebar.number_input(
"공정 게인 K", min_value=1.0, max_value=50.0, value=16.0, step=1.0
)

시간 지연 theta
delay_time = st.sidebar.number_input(
"시간 지연 θ (s)", min_value=0.0, max_value=200.0, value=30.0, step=1.0
)

초기/외기 온도
T_ambient = st.sidebar.number_input(
"초기/외기 온도 (℃)", min_value=0.0, max_value=100.0, value=25.0, step=1.0
)

st.sidebar.markdown("---")
st.sidebar.header("2. PID 파라미터 튜닝")

비례 게인 Kc
Kc = st.sidebar.number_input(
"비례 게인 Kc",
min_value=0.001,
max_value=5.0,
value=0.3,
step=0.01,
format="%.3f",
)

적분 시간 tau_I
tau_I = st.sidebar.number_input(
"적분 시간 τI (s)", min_value=1.0, max_value=2000.0, value=200.0, step=5.0
)

미분 시간 tau_D
tau_D = st.sidebar.number_input(
"미분 시간 τD (s)", min_value=0.0, max_value=200.0, value=25.0, step=1.0
)

st.sidebar.markdown("---")
st.sidebar.header("3. 그래프 표시 설정 (ON / OFF)")
show_temp_graph = st.sidebar.checkbox("온도 응답 곡선 표시", value=True)
show_heater_graph = st.sidebar.checkbox("히터 조작량(u) 곡선 표시", value=True)

---------------------------------------------------------
시뮬레이션 계산 (오일러법)
---------------------------------------------------------
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

성능 지표
final_error = setpoint - T[-1]
overshoot = max(0.0, np.max(T) - setpoint) if np.max(T) > setpoint else 0.0

---------------------------------------------------------
결과 화면 및 실시간 수치 카드 출력
---------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("최종 도달 온도", f"{T[-1]:.2f} ℃")
col2.metric("최종 오차 (Final Error)", f"{final_error:.2f} ℃")
col3.metric("최대 오버슈트 (Overshoot)", f"{overshoot:.2f} ℃")

---------------------------------------------------------
그래프 온오프 렌더링
---------------------------------------------------------
active_plots = sum([show_temp_graph, show_heater_graph])

if active_plots == 0:
st.info("좌측 사이드바에서 최소 하나 이상의 그래프를 켜주세요.")
else:
fig, axes = plt.subplots(
active_plots,
1,
figsize=(10, 4.5 * active_plots),
squeeze=False,
sharex=True,
)
current_ax_idx = 0

# 1) 온도 응답 그래프 (ON일 때만 표시)
if show_temp_graph:
    ax_temp = axes[current_ax_idx, 0]
    ax_temp.plot(
        time,
        T,
        "m-",
        linewidth=2,
        label=f"PID 제어 온도 (최종값: {T[-1]:.2f}℃)",
    )
    ax_temp.axhline(
        setpoint,
        color="r",
        linestyle="--",
        linewidth=1.2,
        label=f"목표치 ({setpoint}℃)",
    )
    ax_temp.set_ylabel("온도 (℃)")
    ax_temp.set_title("시간에 따른 공정 온도 응답")
    ax_temp.grid(True, linestyle=":", alpha=0.6)
    ax_temp.legend(loc="lower right")
    current_ax_idx += 1

# 2) 히터 조작량 그래프 (ON일 때만 표시)
if show_heater_graph:
    ax_heater = axes[current_ax_idx, 0]
    ax_heater.plot(time, u, "g-", linewidth=1.5, label="히터 출력 u (%)")
    ax_heater.set_ylabel("출력 (%)")
    ax_heater.set_ylim([-5, 105])
    ax_heater.set_title("시간에 따른 히터 제어 출력")
    ax_heater.grid(True, linestyle=":", alpha=0.6)
    ax_heater.legend(loc="upper right")

axes[-1, 0].set_xlabel("시간 (s)")
plt.tight_layout()
st.pyplot(fig)
