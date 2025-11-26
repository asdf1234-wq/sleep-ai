import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------
# AI 모델 훈련 및 캐싱 (수면 단계 예측 및 패턴 분석)
# -----------------------------------------------------

@st.cache_resource
def train_sleep_model():
    # 학습 데이터 정의: AI가 깊은 수면 비율에 영향을 미치는 요인 파악을 시뮬레이션
    # 다중 데이터 통합: 카페인, 운동, HR (생체 리듬), 온도, 소음 (환경 데이터)
    data = {
        'Caffeine_mg': [0, 50, 200, 300, 10, 150, 0, 100, 20, 250, 75, 120, 10, 280, 50, 100, 10, 200, 5, 290, 15, 25, 60, 180, 220, 30, 40, 130, 90, 210], 
        'Exercise_intensity': [1, 3, 0, 5, 0, 2, 4, 1, 2, 4, 3, 1, 0, 5, 2, 0, 3, 5, 1, 4, 2, 0, 4, 1, 3, 0, 2, 5, 1, 4], # 0(없음) ~ 5(고강도)
        'Avg_HR': [55, 65, 78, 90, 60, 75, 58, 70, 62, 85, 68, 72, 56, 88, 64, 71, 57, 80, 59, 83, 61, 63, 66, 76, 82, 54, 67, 79, 73, 81], # 취침 전 평균 심박수
        'Temp_C': [19, 22, 25, 26, 18, 23, 20, 21, 17, 24, 20, 22, 18, 25, 21, 23, 19, 24, 17, 26, 20, 22, 23, 24, 25, 16, 21, 26, 22, 23], 
        'Noise_dB': [30, 40, 65, 75, 35, 55, 45, 50, 32, 70, 52, 60, 31, 80, 42, 58, 38, 68, 33, 72, 48, 51, 62, 53, 64, 34, 46, 78, 56, 74],
        
        # 목표 변수: 깊은 수면 비율 (%).
        'Deep_Sleep_Ratio': [35, 20, 10, 5, 18, 25, 28, 15, 33, 12, 22, 17, 31, 8, 26, 14, 32, 7, 29, 6, 21, 16, 27, 13, 9, 34, 30, 11, 23, 19] 
    }
    df = pd.DataFrame(data)

    X = df[['Caffeine_mg', 'Exercise_intensity', 'Temp_C', 'Noise_dB', 'Avg_HR']]
    y = df['Deep_Sleep_Ratio']

    # AI 모델 훈련
    model = LinearRegression()
    model.fit(X, y)
    
    st.info("🤖 AI 수면 패턴 분석 모델 준비 완료.")
    return model

sleep_model = train_sleep_model()

# -----------------------------------------------------
# Streamlit UI 및 실시간 예측 로직
# -----------------------------------------------------
st.set_page_config(layout="wide", page_title="Sleep AI Coach")
st.title("🌙 AI 수면 패턴 최적화 코치")
st.caption("개인의 생체 리듬 및 환경 데이터 기반 실시간 분석 및 맞춤형 코칭 시뮬레이션")
st.markdown("---")

st.subheader("📊 실시간 다중 데이터 입력 (잠자리에 들기 1시간 전 기준)")

# 1. 사용자 입력 (슬라이더) - 실시간으로 앱 재실행 및 예측
col1, col2 = st.columns(2)
with col1:
    caffeine = st.slider("☕️ 카페인 섭취량 (mg)", min_value=0, max_value=300, value=100, step=10, key="caffeine_input")
    exercise = st.slider("🏃‍♂️ 취침 전 운동 강도 (0:없음 ~ 5:고강도)", min_value=0, max_value=5, value=2, step=1, key="exercise_input")
    avg_hr = st.slider("❤️ 취침 전 평균 심박수 (BPM)", min_value=50, max_value=100, value=70, step=1, key="hr_input")

with col2:
    temp = st.slider("🌡️ 침실 온도 (섭씨)", min_value=15, max_value=28, value=22, step=1, key="temp_input")
    noise = st.slider("🔊 침실 소음 레벨 (dB)", min_value=30, max_value=80, value=50, step=5, key="noise_input")
    # (움직임 데이터는 HR/운동 강도에 통합하여 시뮬레이션)
    
st.markdown("---")

# 2. 실시간 AI 예측 실행
input_data = np.array([[caffeine, exercise, temp, noise, avg_hr]])
predicted_ratio = sleep_model.predict(input_data)[0]
final_ratio = max(5.0, min(35.0, round(predicted_ratio, 1))) 

# --- 이 부분을 추가 ---
st.caption(f"디버깅 정보 - AI 모델 원시 예측값: {predicted_ratio:.8f}%")
# --------------------

final_ratio = max(5.0, min(35.0, round(predicted_ratio, 1)))

# -----------------------------------------------------
# 3. AI 분석 및 맞춤형 개입 출력
# -----------------------------------------------------

st.subheader("💡 AI 분석 및 맞춤형 개입 결과 (실시간 코칭)")

col_ratio, col_inter = st.columns([1, 2])

with col_ratio:
    # 수면 단계 예측 및 패턴 분석: 깊은 수면 비율을 높이는 요인 파악
    st.metric(label="예상 깊은 수면 비율", value=f"{final_ratio:.1f}%")
    if final_ratio >= 25.0: st.success("✅ **최적 패턴:** 충분한 회복 수면이 예상됩니다.")
    elif final_ratio >= 15.0: st.warning("⚠️ **개선 필요:** 효율성을 높여야 합니다.")
    else: st.error("❌ **심각한 저하:** 수면의 질이 크게 떨어집니다.")
    
    # 사용자의 개개인의 생체 리듬 분석
    if avg_hr > 75:
        st.markdown(f"**❤️ 생체 리듬 분석:** 심박수({avg_hr}BPM)가 높은 상태입니다. **이완 부족**이 주요 저해 요인입니다.")
    else:
        st.markdown(f"**❤️ 생체 리듬 분석:** 심박수가 안정적({avg_hr}BPM)입니다. 수면 준비가 잘되고 있습니다.")

with col_inter:
    # 맞춤형 개입: 환경 데이터를 분석하여 효과적인 수면 전후 코칭 제공
    intervention_list = ["**AI 추천 최적화 개입 목록 (생활 패턴 적절하게 맞춰줌):**"]
    
    # 환경 개입: 온도 조절, 명상 음악 재생, 조명 색상 변경
    if temp >= 23 or temp <= 18:
        optimal_temp = "19°C~20°C"
        intervention_list.append(f"🌡️ **온도 조절:** 침실 온도를 **{optimal_temp}**로 자동 조절 시도. ({temp}°C는 회복을 방해합니다.)")
    
    if noise >= 50:
        intervention_list.append(f"🔊 **소음/음악:** 백색 소음 또는 **명상 음악**을 재생하여 소음 스트레스를 완화하세요.")
    
    if final_ratio < 20.0:
        intervention_list.append("💡 **조명:** 멜라토닌 분비를 돕는 **붉은 계열의 조명**으로 변경을 추천합니다.")

    # 일상 활동 개입: 카페인 섭취, 운동 시간 조정
    if caffeine >= 150:
        intervention_list.append(f"☕️ **카페인:** 릴랙싱 **허브차**를 마시거나 이완 운동을 통해 각성 효과를 상쇄하세요.")
    
    if exercise >= 4:
        intervention_list.append("🏃‍♂️ **운동:** 취침 1시간 전 고강도 운동은 피하고 **5분 호흡 명상**을 통해 심박수를 낮추세요.")
    elif exercise < 1 and final_ratio < 25.0:
        intervention_list.append("🏃‍♂️ **활동 부족:** 수면의 깊이를 위해 **낮 시간대 규칙적인 유산소 운동**을 생활 패턴에 추가하세요.")

    st.markdown('\n- ' + '\n- '.join(intervention_list))
