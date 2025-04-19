import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Load the trained model
model = joblib.load('cognitive_model2.pkl')
scaler = joblib.load('scaler.pkl')  # 新增这一行

# Define user input fields
st.title("认知能力评估")
age = st.number_input("年龄", min_value=18, max_value=80, step=1)
gender = st.radio("性别", ["男", "女", "不愿透露"])
sleep_duration = st.number_input("睡眠时间 (小时<=24)", min_value=0.0, max_value=24.0, step=0.1)
stress_level = st.slider("压力水平 (1-10)", min_value=1, max_value=10)
diet_type = st.radio("饮食习惯", ["非素食主义者", "纯素主义者", "素食主义者"])
screen_time = st.number_input("日常屏幕时间 (小时<=24)", min_value=0.0, max_value=24.0, step=0.1)
exercise = st.radio("运动频率", ["低", "中等", "高"])
caffeine = st.number_input("咖啡因摄入量 (mg<=500)", min_value=0, max_value=500, step=1)
reaction_time = st.number_input("反应时间 (ms<=600)", min_value=0.0, max_value=600.0, step=0.1)
memory_score = st.number_input("记忆测试分数", min_value=0, max_value=100, step=1)

# Predict button
if st.button("开始评估"):
    # Create a dictionary with user inputs
    input_dict = {
        'Age': age,
        'Sleep_Duration': sleep_duration,
        'Stress_Level': stress_level,
        'Daily_Screen_Time': screen_time,
        'Caffeine_Intake': caffeine,
        'Reaction_Time': reaction_time,
        'Memory_Test_Score': memory_score,
        'Gender_Female': 1 if gender == "女" else 0,
        'Gender_Male': 1 if gender == "男" else 0,
        'Gender_Other': 1 if gender == "不愿透露" else 0,  
        'Diet_Type_Non-Vegetarian': 1 if diet_type == "非素食主义者" else 0,
        'Diet_Type_Vegan': 1 if diet_type == "纯素主义者" else 0,
        'Diet_Type_Vegetarian': 1 if diet_type == "素食主义者" else 0,
        'Exercise_Frequency_High': 1 if exercise == "高" else 0,
        'Exercise_Frequency_Low': 1 if exercise == "低" else 0,
        'Exercise_Frequency_Medium': 1 if exercise == "中等" else 0
    }

    # Convert to DataFrame to ensure correct feature order
    input_df = pd.DataFrame([input_dict])
    expected_columns = ['Age', 'Sleep_Duration', 'Stress_Level', 'Daily_Screen_Time',
                        'Caffeine_Intake', 'Reaction_Time', 'Memory_Test_Score',
                        'Gender_Female', 'Gender_Male', 'Gender_Other',
                        'Diet_Type_Non-Vegetarian', 'Diet_Type_Vegan', 'Diet_Type_Vegetarian',
                        'Exercise_Frequency_High', 'Exercise_Frequency_Low', 'Exercise_Frequency_Medium']
    input_df = input_df[expected_columns]
    input_scaled = scaler.transform(input_df)  # 新增
    # Make prediction
    prediction = model.predict(input_scaled)

    # Visualization of prediction
    st.write("## 评估分析")
    col1, col2 = st.columns(2)
    if prediction[0] < 1:
        prediction[0] = 1
    elif prediction[0] > 100:
        prediction[0] = 100
        
    col1.metric("## 认知能力评估分数:", f"{prediction[0]:.2f}")

    # Reaction time analysis (strong negative correlation visible in plot)
    if prediction[0] <= 30:
        col2.error("⚠️ 认知衰退风险：高")
    if prediction[0] > 30 and prediction[0] <= 65:
        col2.warning("⚠️ 认知衰退风险：中")
    if prediction[0] > 65:
        col2.success("✅ 认知衰退风险：低")

    # Additional insights based on visualization relationships
    st.write("### 影响您得分的关键因素:")
    factors = []

    # Reaction time analysis (strong negative correlation visible in plot)
    if reaction_time > 500:
        factors.append("⚠️ 您较低的认知分数与较长的反应时间（>500ms）显著相关。")
    elif reaction_time > 400:
        factors.append("⚠️ 您中等的反应时间可能影响认知表现。")

    # Sleep duration impact
    if sleep_duration < 6:
        factors.append("⚠️ 每天睡不够6小时，脑子可能变迟钝！")
        
    # Memory score analysis (based on the banded distribution) 
    if memory_score < 60:
        factors.append("⚠️ 记忆力测试得分低？可能是大脑发出的警告信号！")

    # Stress level impact
    if stress_level > 7:
        factors.append("⚠️ 高压力水平可能损害认知功能。")

    # Screen time analysis
    if screen_time > 7:
        factors.append("⚠️ 长时间使用电子设备可能导致认知疲劳")

        # exercise analysis
    if exercise == "Low":
        factors.append("⚠️ 运动过少可能损害认知功能")


    # Display all factors as a bulleted list
    for factor in factors:
        st.write(factor)

    # Visualization of where this score falls on distribution
    st.write("### 您的分数在人群的位置:")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.linspace(0, 100, 1000)
    y = np.exp(-0.5 * ((x - 50) / 15)**2) 
    ax.plot(x, y)
    ax.axvline(x=prediction[0], color='blue', linestyle='--')
    ax.fill_between(x[x <= prediction[0]], y[x <= prediction[0]], alpha=0.3, color='red')
    ax.set_xlabel('Cognitive Score')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 100)
    ax.set_title('Your Score Relative to Population Distribution')
    st.pyplot(fig)
