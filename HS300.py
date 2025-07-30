# -*- coding: utf-8 -*-
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import akshare as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime

# 1. 指定中文字体路径（相对路径）
font_path = "./fonts/simhei.ttf"  # 替换为你的字体路径

# 2. 动态添加字体
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)

# 3. 全局设置中文字体
plt.rcParams["font.family"] = font_prop.get_name()  # 使用字体名称
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 设置Streamlit页面
st.set_page_config(page_title="沪深300预测系统", layout="wide")
st.title("📈 沪深300指数预测可视化系统")

# 侧边栏参数设置
with st.sidebar:
    st.header("模型参数")
    epochs = st.slider("训练轮次(epochs)", 50, 200, 100)
    timesteps = st.slider("时间步长(timesteps)", 30, 90, 60)
    batch_size = st.slider("批大小(batch_size)", 16, 64, 32)

    st.divider()
    st.info(f"当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if st.button("重新训练模型"):
        st.experimental_rerun()


# 获取数据
@st.cache_data
def load_data():
    hs300 = ak.stock_zh_index_daily(symbol="sh000300")
    hs300['date'] = pd.to_datetime(hs300['date'])
    hs300.set_index('date', inplace=True)
    return hs300[hs300.index >= pd.to_datetime('2020-01-01')].copy()


try:
    hs300 = load_data()
    last_date = hs300.index[-1].date()

    # 计算预测日期（下一个工作日）
    next_trading_day = last_date + pd.offsets.BDay(1)
    st.success(f"数据加载完成! 最后数据日期: {last_date.strftime('%Y-%m-%d')}")
except Exception as e:
    st.error(f"数据加载失败: {str(e)}")
    st.stop()

# 显示数据摘要
st.subheader("历史数据概览")
st.dataframe(hs300.tail().style.format({'close': '{:.2f}', 'volume': '{:,.0f}'}), height=150)
st.line_chart(hs300[['close']], use_container_width=True)

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(hs300[['close']].values)


# 创建时间序列数据集
def create_dataset(data, timesteps):
    X, y = [], []
    for i in range(timesteps, len(data)):
        X.append(data[i - timesteps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


X, y = create_dataset(scaled_data, timesteps)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 训练模型
st.subheader("模型训练过程")
progress_bar = st.progress(0)
status_text = st.empty()

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(timesteps, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 简化训练显示
history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                    validation_split=0.2, verbose=0)

# 显示训练过程
progress_bar.progress(100)
status_text.success("模型训练完成!")

# 绘制损失曲线
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history.history['loss'], label='训练损失')
ax.plot(history.history['val_loss'], label='验证损失')
ax.set_title('模型损失变化')
ax.set_ylabel('损失')
ax.set_xlabel('训练轮次')
ax.legend()
st.pyplot(fig)

# 预测下一个交易日
last_60_days = scaled_data[-timesteps:]
last_60_days = last_60_days.reshape(1, timesteps, 1)
pred_scaled = model.predict(last_60_days)
tomorrow_pred = scaler.inverse_transform(pred_scaled)[0][0]

# 计算历史准确性
predictions = []
for i in range(timesteps, len(scaled_data)):
    input_data = scaled_data[i - timesteps:i].reshape(1, timesteps, 1)
    pred = model.predict(input_data, verbose=0)
    predictions.append(scaler.inverse_transform(pred)[0][0])

actuals = hs300.iloc[timesteps:]['close'].values
errors = np.abs(predictions - actuals)
accuracy = 100 * (1 - errors / actuals)

# 显示结果
st.subheader("预测结果")
pred_color = "green" if tomorrow_pred > hs300['close'].iloc[-1] else "red"
pred_date = next_trading_day.strftime("%Y-%m-%d")

st.metric(label=f"{pred_date} 预测收盘价",
          value=f"{tomorrow_pred:.2f}",
          delta=f"{(tomorrow_pred - hs300['close'].iloc[-1]):.2f}",
          delta_color="normal")

# 可视化预测
st.subheader("近期表现与预测")
fig2, ax2 = plt.subplots(figsize=(12, 6))

# 最后60天实际数据
last_60_days_actual = hs300.iloc[-60:]
ax2.plot(last_60_days_actual.index, last_60_days_actual['close'],
         'b-', label='历史数据')

# 预测点
prediction_point = pd.date_range(last_60_days_actual.index[-1], periods=2, freq='B')[1]
ax2.plot(prediction_point, tomorrow_pred, 'ro', markersize=8, label='预测值')

# 历史准确性（最后30个预测点）
accuracy_df = pd.DataFrame({
    'date': hs300.index[timesteps:][-30:],
    'predicted': predictions[-30:],
    'actual': actuals[-30:]
})
ax2.plot(accuracy_df['date'], accuracy_df['predicted'], 'g--', alpha=0.7, label='历史预测')

# 格式设置
ax2.set_title('沪深300指数预测')
ax2.set_xlabel('日期')
ax2.set_ylabel('收盘价')
ax2.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)

# 模型性能统计
st.subheader("模型性能评估")
col1, col2, col3 = st.columns(3)
col1.metric("平均预测误差", f"{np.mean(errors[-30:]):.2f}")
col2.metric("预测准确性", f"{np.mean(accuracy[-30:]):.2f}%")
col3.metric("最新实际数据", f"{hs300['close'].iloc[-1]:.2f}")

# 显示历史预测准确性
st.line_chart(pd.DataFrame({
    '实际值': actuals[-30:],
    '预测值': predictions[-30:]
}), use_container_width=True)
