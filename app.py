import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ------ UI THEME (Light Blue Background) ------
st.markdown("""
<style>
.stApp { background-color: #a9bfd6; }
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > select {
    background-color: #f8fcff !important;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------ AUTO-GENERATED DATASET (120 samples) ------
num_samples = 120
distance = np.random.uniform(0.5, 7, num_samples)
slope = np.random.randint(1, 4, num_samples)
terrain = np.random.choice(["road", "grass", "stairs"], num_samples)
speed = np.random.uniform(3, 7, num_samples)

df = pd.DataFrame({
    "distance_km": distance,
    "slope_level": slope,
    "terrain": terrain,
    "speed_kmph": speed
})

# ------ STEP & CALORIE FORMULAS ------
df["steps"] = (df["distance_km"] * 1200) + (df["slope_level"] * 140) + (df["speed_kmph"] * 8)
df["calories"] = (df["distance_km"] * 55) + (df["speed_kmph"] * 10) + (df["slope_level"] * 12)

# ------ LABEL ENCODING FOR TERRAIN ------
le = LabelEncoder()
df["terrain"] = le.fit_transform(df["terrain"])

# ------ FEATURES & TARGETS ------
X = df[["distance_km", "slope_level", "terrain", "speed_kmph"]]
y_steps = df["steps"]
y_cal = df["calories"]

# ------ FEATURE SCALING ------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------ MACHINE LEARNING MODELS ------
model_steps = RandomForestRegressor(n_estimators=200, random_state=42)
model_cal = RandomForestRegressor(n_estimators=200, random_state=42)

model_steps.fit(X_scaled, y_steps)
model_cal.fit(X_scaled, y_cal)

# ------ STREAMLIT UI ------
st.title("🏃‍♂️ FitRoute AI - Step & Calorie Predictor")
st.write("Enter your route details below:")

distance = st.number_input("Distance (km)", 0.5, 20.0, 3.0)
slope = st.selectbox("Slope Level", [1, 2, 3])
terrain_name = st.selectbox("Terrain Type", ["road", "grass", "stairs"])
speed = st.number_input("Walking Speed (km/h)", 3.0, 7.0, 5.0)

# ------ PREDICTION ------
if st.button("Predict"):
    terrain_encoded = le.transform([terrain_name])[0]
    new_data = np.array([[distance, slope, terrain_encoded, speed]])
    new_scaled = scaler.transform(new_data)

    pred_steps = int(model_steps.predict(new_scaled)[0])
    pred_cal = model_cal.predict(new_scaled)[0]

    st.success(f"Estimated Steps: {pred_steps}")
    st.info(f"Estimated Calories Burned: {round(pred_cal, 2)}")

# ------ SUMMARY TABLE ------
summary_df = pd.DataFrame({
    "Route Type": ["Road", "Grass", "Stairs"],
    "Average Steps": [1200, 1450, 1700],
    "Average Calories": [60, 75, 95]
})

st.subheader("📊 Route Type Summary Table")
st.table(summary_df)

# ------ GRAPH: Steps vs Calories ------
fig, ax = plt.subplots()
ax.scatter(df["steps"], df["calories"], color="red", alpha=0.7)
ax.set_xlabel("Steps")
ax.set_ylabel("Calories")
ax.set_title("Steps vs Calories Graph")

st.subheader("📈 Steps vs Calories Graph")
st.pyplot(fig)
