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
    demo['discount_rate'] = np.random.uniform(0, 0.3, size=30)
    demo['price'] = demo['price'] * (1 - demo['discount_rate'])
    demo['sales_boost_proxy'] = (
    300 * demo['discount_rate'] +
    np.random.normal(0, 30, size=30) -
    demo['price'] * 0.05
)
    - demo['price'] * 0.05
)
demo['sales_boost_rate'] = demo['sales_boost_proxy'] / demo['base_sales']
cce = demo.groupby('streamer_id')['sales_boost_rate'].mean().reset_index()
cce.columns = ['streamer_id', 'CCE']
demo = demo.merge(cce, on='streamer_id')
demo['actual_sales'] = demo['base_sales'] + demo['sales_boost_proxy'] + demo['CCE'] * 100
    demo['sales_boost_rate'] = (demo['actual_sales'] - demo['base_sales']) / demo['base_sales']
    cce = demo.groupby('streamer_id')['sales_boost_rate'].mean().reset_index()
    cce.columns = ['streamer_id', 'CCE']
    return demo.merge(cce, on='streamer_id')

# 模型培训
def train_model(data):
    X = data[['price', 'CCE', 'discount_rate']]
    y = data['actual_sales']
    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05)
    model.fit(X, y)
    return model

# 市场份额函数（logit based）
def calculate_share(p1, p2, alpha=0.01):
    exp1 = np.exp(-alpha * p1)
    exp2 = np.exp(-alpha * p2)
    total = exp1 + exp2
    return exp1 / total, exp2 / total

# Streamlit UI
st.title("AI Agent for Smartphone Sales Forecast with Influencer Impact")

data = load_data()
model = train_model(data)

st.sidebar.header("Input Product Parameters")
price = st.sidebar.number_input("Price ($)", min_value=100.0, max_value=3000.0, value=999.0, step=1.0)
discount = st.sidebar.slider("Discount Rate (%)", min_value=0, max_value=50, value=10) / 100
brand = st.sidebar.selectbox("Select Phone Brand", data['brand'].unique())
streamer = st.sidebar.selectbox("Choose Influencer", data['streamer_id'].unique())

cce_value = data[data['streamer_id'] == streamer]['CCE'].iloc[0]
adjusted_price = price * (1 - discount)

X_new = pd.DataFrame([[adjusted_price, cce_value, discount]], columns=['price', 'CCE', 'discount_rate'])
pred_sales = model.predict(X_new)[0]

st.subheader("📈 Predicted Sales")
st.caption("""
This section uses a Gradient Boosting Regressor trained on historical smartphone sales and influencer performance. 
Features include price, discount rate, and the Celebrity Coefficient (CCE), which simulates influencer impact. 
The output reflects a 6-month sales forecast under given conditions.
""")
st.metric(label="Expected 6-month Sales", value=f"{int(pred_sales)} units")

# 市场敏感月度预测图表
st.subheader("📊 Seasonal E-commerce Sales Forecast (Peak Season View)")
st.caption("""
This chart reflects expected monthly sales trends, adjusted for North American holiday patterns (Sep–Mar). 
It combines predefined seasonal multipliers with discount-sensitive late-stage boosts to simulate realistic market curves.
""")
holiday_boost = [1.6, 1.4, 1.1, 1.2, 0.8, 0.7] if 9 <= pd.Timestamp.today().month or pd.Timestamp.today().month <= 3 else [1.5, 1.2, 0.9, 1.3, 1.0 + discount * 1.5, 1.0 + discount * 2.0]
monthly_weights = holiday_boost  # 更贴近现实的非线性波动
monthly_sales = (pred_sales * np.array(monthly_weights)).astype(int)
months = [f"Month {i+1}" for i in range(6)]
monthly_df = pd.DataFrame({"Month": months, "Sales": monthly_sales})
# Add realistic North American holidays for Sep–Mar season
holiday_labels = [
    "📚 Back to School",  # Month 1
    "🍂 Thanksgiving",     # Month 2
    "🛍️ Black Friday",     # Month 3
    "🎄 Christmas",        # Month 4
    "🎉 New Year",         # Month 5
    "🏈 Super Bowl"        # Month 6
] if 9 <= pd.Timestamp.today().month or pd.Timestamp.today().month <= 3 else ["" for _ in range(6)]
monthly_df["Holiday"] = holiday_labels
line_fig = px.line(monthly_df, x="Month", y="Sales", text="Holiday", markers=True, title="📆 Seasonal Sales Curve with North American Holidays")
st.plotly_chart(line_fig)

# 品牌+主播比较柱状图
st.subheader("🔍 Brand & Influencer Comparison")
st.caption("""
This bar chart compares predicted sales under different brand and influencer pairings. 
Each influencer is evaluated using their CCE (Celebrity Coefficient), which is derived from historical uplift rates.
""")
predictions = []
for b in data['brand'].unique():
    for s in data['streamer_id'].unique():
        cce_val = data[data['streamer_id'] == s]['CCE'].iloc[0]
        pred = model.predict(pd.DataFrame([[adjusted_price, cce_val, discount]], columns=['price', 'CCE', 'discount_rate']))[0]
        predictions.append({'Brand': b, 'Influencer': s, 'Predicted Sales': pred})

report_df = pd.DataFrame(predictions)
fig = px.bar(report_df, x='Influencer', y='Predicted Sales', color='Brand', barmode='group', title="Sales Forecast by Brand & Influencer")
st.plotly_chart(fig)

# 利润模拟器：基于价格决定市场份额与利润
st.subheader("📊 Dual Product Profit Simulator")
st.caption("""
This module estimates optimal profit using a logit-based share model, where prices determine expected market share. 
It calculates revenue, cost, and profit based on price and marginal cost inputs, and provides a strategic suggestion.
""")
price1 = st.number_input("Phone S1 Price", value=799)
price2 = st.number_input("Phone S2 Price", value=899)
mc1 = st.number_input("S1 Marginal Cost", value=440)
mc2 = st.number_input("S2 Marginal Cost", value=470)
share1, share2 = calculate_share(price1, price2)
M = 10000  # 市场容量
q1 = share1 * M
q2 = share2 * M
rev1 = q1 * price1
rev2 = q2 * price2
cost1 = mc1 * q1
cost2 = mc2 * q2
profit1 = rev1 - cost1
profit2 = rev2 - cost2
total_profit = profit1 + profit2

profit_df = pd.DataFrame({
    "Phone": ["S1", "S2"],
    "Price": [price1, price2],
    "Market Share": [share1, share2],
    "Quantity": [q1, q2],
    "Revenue": [rev1, rev2],
    "Cost": [cost1, cost2],
    "Profit": [profit1, profit2]
})

st.dataframe(profit_df.round(2))
st.write(f"**Total Estimated Profit:** ${total_profit:.2f}")

# 提示最佳策略（基于利润差异）
if profit1 > profit2:
    suggestion = "📌 Consider increasing focus on S1 — higher price-performance profit observed."
elif profit2 > profit1:
    suggestion = "📌 S2 pricing strategy currently yields more profit — optimize cost or promote aggressively."
else:
    suggestion = "⚖️ Both products yield equal profit — fine-tune market strategy."

st.info(suggestion)

st.subheader("📌 Celebrity Coefficient Table")
st.caption("""
This table lists each influencer's average impact on historical sales. 
The CCE score is used as an input feature for forecasting sales uplift.
""")
st.dataframe(data[['streamer_id', 'CCE']].drop_duplicates().reset_index(drop=True))
