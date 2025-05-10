
# 📱 AI Agent: Smartphone Sales Forecast & Profit Optimization

A Streamlit-based AI platform that simulates smartphone sales trends, pricing impact, influencer contribution, and seasonal marketing outcomes across North America.

---

## 🚀 Features

### 📈 1. Predicted Sales
- Uses `GradientBoostingRegressor` on features: **price, discount rate, Celebrity Coefficient (CCE)**.
- Outputs **6-month forecasted sales** for a given product-influencer combination.

### 📊 2. Seasonal E-commerce Sales Forecast
- Adds **North American holiday awareness** (Back to School, Black Friday, Christmas, etc.).
- Applies **dynamic month-to-month multipliers** adjusted for peak season and discount rates.

### 🔍 3. Brand & Influencer Comparison
- Compares sales predictions under multiple brand and influencer pairings.
- Based on historical influencer performance (`CCE`).

### 💸 4. Dual Product Profit Simulator
- Lets user input **price** and **marginal cost** for two competing phones (S1 & S2).
- Simulates market share using a **logit function**.
- Computes **quantity, revenue, cost, profit**, and **offers pricing strategy suggestions**.

### 📌 5. Celebrity Coefficient Table
- Displays each influencer's historical contribution to sales uplift.
- CCE is used directly in the forecasting model.

---

## 📊 Sample Use Cases
- Optimize pricing strategy during seasonal peaks.
- Forecast revenue impact of different influencers.
- Compare profitability between flagship and mid-range products.
- Simulate pricing wars between two models under cost constraints.

---

## 🧠 Algorithm Insights
- `GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05)`
- Market share: `exp(-α * price) / sum of exponentials` with α = 0.01
- Monthly weights dynamically adjust based on:
  - Current calendar (Sep–Mar as peak)
  - Holiday labels (🎄 Christmas, 🛍️ Black Friday, etc.)
  - Discount elasticity

---

## 🔧 Technologies Used
- Python, Streamlit
- Scikit-learn, Pandas, NumPy
- Plotly for visualizations
