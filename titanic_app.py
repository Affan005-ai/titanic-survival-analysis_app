import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Load Titanic dataset
df = pd.read_csv("titanic.csv")

# Preprocessing
df['age'].fillna(df['age'].mean(), inplace=True)
df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 60, 120], labels=['Child', 'Teen', 'Adult', 'Senior'])

# Sidebar filters
st.sidebar.title("Titanic Data Filters")
selected_gender = st.sidebar.selectbox("Select Gender:", ['all'] + sorted(df['sex'].dropna().unique().tolist()))
selected_class = st.sidebar.selectbox("Select Passenger Class:", ['all'] + sorted(df['pclass'].dropna().unique().tolist()))

# Apply filters
filtered_df = df.copy()
if selected_gender != 'all':
    filtered_df = filtered_df[filtered_df['sex'] == selected_gender]
if selected_class != 'all':
    filtered_df = filtered_df[filtered_df['pclass'] == selected_class]

# Display data summary
st.title("🚢 Titanic Survival Analysis")
st.write("### Filtered Data Snapshot")
st.dataframe(filtered_df.head())

# Grouped survival rate
st.write("### Survival Rate by Gender")
survival_gender = filtered_df.groupby('sex')['survived'].mean().reset_index()
st.bar_chart(survival_gender.set_index('sex'))

st.write("### Average Fare by Class")
fare_by_class = filtered_df.groupby('pclass')['fare'].mean().reset_index()
st.bar_chart(fare_by_class.set_index('pclass'))

# One-hot encoding display
st.write("### One-Hot Encoding Example")
dummies = pd.get_dummies(filtered_df['sex'], drop_first=True)
st.dataframe(dummies.head())

# Custom survival prediction
st.write("### Predict Survival Probability")
gender_input = st.selectbox("Select Gender for Prediction", ['male', 'female'])
class_input = st.selectbox("Select Class for Prediction", [1, 2, 3])

# Survival model from full dataset
combined_probs = df.groupby(['sex', 'pclass'])['survived'].mean().unstack()
try:
    prob = combined_probs.loc[gender_input.lower(), class_input]
    st.success(f"Predicted survival chance: {prob:.2f} or {prob*100:.1f}%")
except:
    st.error("Invalid input for prediction.")
