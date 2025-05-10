import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import plotly.express as px

# 数据初始化
def load_data():
    df = pd.read_csv("smartphone_customer_data.csv")
    np.random.seed(42)
    demo = df.head(30).copy()
    streamers = ['H1', 'H2', 'S1', 'S2']
    brands = ['huawei', 'apple', 'samsung']
    demo['streamer_id'] = np.random.choice(streamers, size=30)
    demo['brand'] = np.random.choice(brands, size=30)
    demo['base_sales'] = np.random.randint(500, 1000, size=30)
    demo['actual_sales'] = demo['base_sales'] + np.random.randint(50, 300, size=30)
    demo['sales_boost_rate'] = (demo['actual_sales'] - demo['base_sales']) / demo['base_sales']
    cce = demo.groupby('streamer_id')['sales_boost_rate'].mean().reset_index()
    cce.columns = ['streamer_id', 'CCE']
    return demo.merge(cce, on='streamer_id')

# 模型培训
def train_model(data):
    X = data[['price', 'CCE']]
    y = data['actual_sales']
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model

# Streamlit UI
st.title("AI Agent for Smartphone Sales Forecast with Influencer Impact")

data = load_data()
model = train_model(data)

st.sidebar.header("Input Product Parameters")
price = st.sidebar.number_input("Price ($)", min_value=100.0, max_value=3000.0, value=999.0, step=1.0)
brand = st.sidebar.selectbox("Select Phone Brand", data['brand'].unique())
streamer = st.sidebar.selectbox("Choose Influencer", data['streamer_id'].unique())

cce_value = data[data['streamer_id'] == streamer]['CCE'].iloc[0]

X_new = pd.DataFrame([[price, cce_value]], columns=['price', 'CCE'])
pred_sales = model.predict(X_new)[0]

st.subheader("📈 Predicted Sales")
st.metric(label="Expected 6-month Sales", value=f"{int(pred_sales)} units")

# 生成可视化报告
st.subheader("🔍 Brand & Influencer Comparison")
predictions = []
for b in data['brand'].unique():
    for s in data['streamer_id'].unique():
        cce_val = data[data['streamer_id'] == s]['CCE'].iloc[0]
        pred = model.predict(pd.DataFrame([[price, cce_val]], columns=['price', 'CCE']))[0]
        predictions.append({'Brand': b, 'Influencer': s, 'Predicted Sales': pred})

report_df = pd.DataFrame(predictions)
fig = px.bar(report_df, x='Influencer', y='Predicted Sales', color='Brand', barmode='group')
st.plotly_chart(fig)

st.subheader("📌 Celebrity Coefficient Table")
st.dataframe(data[['streamer_id', 'CCE']].drop_duplicates().reset_index(drop=True))
