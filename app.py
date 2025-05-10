import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# 数据初始化
def load_data():
    df = pd.read_csv("smartphone_customer_data.csv")
    np.random.seed(42)
    demo = df.head(30).copy()
    streamers = ['H1', 'H2', 'S1', 'S2']
    demo['streamer_id'] = np.random.choice(streamers, size=30)
    demo['base_sales'] = np.random.randint(500, 1000, size=30)
    demo['actual_sales'] = demo['base_sales'] + np.random.randint(50, 300, size=30)
    demo['sales_boost_rate'] = (demo['actual_sales'] - demo['base_sales']) / demo['base_sales']
    cce = demo.groupby('streamer_id')['sales_boost_rate'].mean().reset_index()
    cce.columns = ['streamer_id', 'CCE']
    return demo.merge(cce, on='streamer_id')

# 模型训练
def train_model(data):
    X = data[['price', 'screen_size', 'CCE']]
    y = data['actual_sales']
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model

# Streamlit UI
st.title("AI Agent for Smartphone Sales Forecast with Influencer Impact")

data = load_data()
model = train_model(data)

st.sidebar.header("Input New Product Parameters")
price = st.sidebar.slider("Price ($)", 400, 1200, 799)
screen_size = st.sidebar.slider("Screen Size (inches)", 4.5, 7.0, 5.7)
streamer = st.sidebar.selectbox("Choose Streamer", ['H1', 'H2', 'S1', 'S2'])

cce_value = data[data['streamer_id'] == streamer]['CCE'].iloc[0]

X_new = pd.DataFrame([[price, screen_size, cce_value]], columns=['price', 'screen_size', 'CCE'])
pred_sales = model.predict(X_new)[0]

st.subheader("Predicted Sales")
st.metric(label="Expected 6-month Sales", value=f"{int(pred_sales)} units")

st.write("\n\n")
st.subheader("Feature Insights")
st.write("Using CCE (Celebrity Coefficient) to simulate the impact of live-streaming influencers on smartphone sales.")
st.dataframe(data[['streamer_id', 'CCE']].drop_duplicates().reset_index(drop=True))