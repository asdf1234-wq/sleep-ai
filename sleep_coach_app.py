import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- AI 모델 훈련 및 캐싱 (수면 단계 예측 및 패턴 분석 시뮬레이션) ---
@st.cache_resource
def train_sleep_model():
    # Feature: 카페인, 운동 강도, 온도, 소음 레벨, 평균 심박수 (다중 데이터 통합 시뮬레이션)
    # Target: 깊은 수면 비율 (%)
    data = {
        'Caffeine_mg': [50, 200, 0, 150, 10, 300, 50, 100], 
        'Exercise_intensity': [3, 0, 1, 5, 0, 2, 4, 1],
        'Temp_C': [22, 25, 18, 20, 23, 26, 19, 21], 
        'Noise_dB': [40, 65, 30, 55, 45, 70, 35, 50],
        'Avg_HR': [65, 78, 58, 70, 68, 85, 62, 75],
        'Deep_Sleep_Ratio': [20, 10, 30, 25, 18, 5, 28, 15] 
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

# --- Streamlit UI 및 맞춤형 개입 (Coaching) 로직 ---
st.set_page_config(layout="wide", page_title="Sleep AI Coach")
st.title("🌙 AI 수면 패턴 최적화 코치")
st.caption("웨어러블 및 환경 데이터 기반 맞춤형 개입 시뮬레이션 (실시간 분석)")
st.markdown("---")

st.subheader("📊 다중 데이터 입력 (잠자리에 들기 1시간 전 기준)")

# 1. 사용자 입력 (슬라이더)
col1, col2 = st.columns(2)
with col1:
    caffeine = st.slider("☕️ 카페인 섭취량 (mg)", min_value=0, max_value=300, value=100, step=10, key="caffeine_input")
    exercise = st.slider("🏃‍♂️ 취침 전 운동 강도 (0:없음 ~ 5:고강도)", min_value=0, max_value=5, value=2, step=1, key="exercise_input")

with col2:
    temp = st.slider("🌡️ 침실 온도 (섭씨)", min_value=15, max_value=28, value=22, step=1, key="temp_input")
    noise = st.slider("🔊 침실 소음 레벨 (dB)", min_value=30, max_value=80, value=50, step=5, key="noise_input")
    avg_hr = st.slider("❤️ 취침 전 평균 심박수 (BPM)", min_value=50, max_value=100, value=70, step=1, key="hr_input")

st.markdown("---")

# 2. 실시간 AI 예측 및 분석 (버튼 없이 즉시 실행)
# st.spinner는 Streamlit Cloud에서만 잘 작동하므로, 실시간 업데이트를 위해 제거하거나 단순화합니다.

# 1. 수면 단계 예측 및 패턴 분석
input_data = np.array([[caffeine, exercise, temp, noise, avg_hr]])
predicted_ratio = sleep_model.predict(input_data)[0]
final_ratio = max(5.0, min(35.0, round(predicted_ratio, 1))) # 비율은 5%~35%로 제한

st.subheader("💡 AI 분석 및 맞춤형 개입 결과")

col_ratio, col_inter = st.columns([1, 2])

with col_ratio:
    # 예상 깊은 수면 비율 출력 및 상태 표시
    st.metric(label="예상 깊은 수면 비율", value=f"{final_ratio:.1f}%")
    if final_ratio >= 25.0: st.success("✅ 최적의 패턴 예상")
    elif final_ratio >= 15.0: st.warning("⚠️ 개선 필요")
    else: st.error("❌ 심각한 저하 예상")
    
    # 심박수 피드백
    if avg_hr > 75:
        st.markdown(f"**심박수 분석:** 높은 편입니다. 이완이 부족합니다.")

with col_inter:
    # 3. 맞춤형 개입 (코칭 메시지)
    intervention_list = ["**AI 추천 최적화 개입 목록:**"]
    
    # 환경 개입: 온도, 소음, 조명/음악 추천
    if temp >= 23 or temp <= 18:
        intervention_list.append(f"🌡️ **온도:** 침실 온도를 **19°C**로 자동 조절을 시도하세요.")
    if noise >= 50:
        intervention_list.append(f"🔊 **소음:** 백색 소음 또는 **명상 음악** 재생을 추천합니다.")
    
    # Deep Sleep 비율이 낮을 때 조명 개입
    if final_ratio < 20.0:
        intervention_list.append("💡 **조명:** 멜라토닌 분비를 돕는 **붉은 계열 조명**으로 변경하세요.")

    # 활동/섭취 개입
    if caffeine >= 150:
        intervention_list.append(f"☕️ **카페인:** 릴랙싱 **허브차**를 마시며 심박수를 낮추세요.")
    if exercise >= 4:
        intervention_list.append("🏃‍♂️ **운동:** 취침 1시간 전에는 **5분 호흡 명상**을 통해 심박수를 낮추세요.")

    st.markdown('\n- ' + '\n- '.join(intervention_list))
