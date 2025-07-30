# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 启用GPU加速

import akshare as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
import matplotlib.font_manager as fm
import datetime
import time
import psutil
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ========== 系统初始化 ==========
# 1. 指定中文字体路径
font_path = "./fonts/simhei.ttf"  # 确保字体文件存在

# 2. 动态添加字体
try:
    font_prop = fm.FontProperties(fname=font_path)
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
except Exception as e:
    st.warning(f"字体加载失败: {str(e)}，使用默认字体")

# 3. GPU内存优化配置
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        st.error(f"GPU配置错误: {str(e)}")

# 4. 设置Streamlit页面
st.set_page_config(page_title="沪深300预测系统", layout="wide")
st.title("📈 沪深300指数预测可视化系统")

# ========== 参数设置 ==========
with st.sidebar:
    st.header("模型参数")
    epochs = st.slider("训练轮次(epochs)", 50, 200, 100)
    timesteps = st.slider("时间步长(timesteps)", 30, 90, 60)
    batch_size = st.slider("批大小(batch_size)", 16, 64, 32)
    
    st.divider()
    st.info(f"当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if st.button("重新训练模型"):
        keys = ['model', 'predictions', 'scaled_data', 'X', 'y']
        for key in keys:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()

# ========== 数据获取与预处理 ==========
@st.cache_data(ttl=3600, show_spinner="加载历史数据...")
def load_data():
    try:
        hs300 = ak.stock_zh_index_daily(symbol="sh000300")
        hs300['date'] = pd.to_datetime(hs300['date'])
        hs300.set_index('date', inplace=True)
        # 数据清洗：处理异常值和缺失值
        hs300 = hs300[hs300['close'] > 0].ffill().bfill()
        return hs300[hs300.index >= pd.to_datetime('2020-01-01')].copy()
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        st.stop()

try:
    with st.spinner('正在加载沪深300历史数据...'):
        hs300 = load_data()
        last_date = hs300.index[-1].date()
        next_trading_day = last_date + pd.offsets.BDay(1)
    st.success(f"数据加载完成! 最后数据日期: {last_date.strftime('%Y-%m-%d')}")
except Exception as e:
    st.error(f"数据初始化失败: {str(e)}")
    st.stop()

# 显示数据摘要
st.subheader("历史数据概览")
st.dataframe(hs300.tail().style.format({'close': '{:.2f}', 'volume': '{:,.0f}'}), height=150)
st.line_chart(hs300[['close']], use_container_width=True)

# 数据预处理（缓存预处理结果）
@st.cache_data
def preprocess_data(_hs300):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(_hs300[['close']].values)
    
    # 数据验证
    if np.isnan(scaled_data).any():
        st.error("数据包含NaN值！")
        st.stop()
    return scaler, scaled_data

scaler, scaled_data = preprocess_data(hs300)

# 创建时间序列数据集
@st.cache_data
def create_dataset(_data, _timesteps):
    X, y = [], []
    for i in range(_timesteps, len(_data)):
        X.append(_data[i - _timesteps:i, 0])
        y.append(_data[i, 0])
    return np.array(X), np.array(y)

# ========== 模型构建与训练 ==========
def build_model(_timesteps):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(_timesteps, 1)),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(_model, _X, _y, _epochs, _batch_size):
    # 动态调整batch_size避免内存溢出
    adjusted_batch = min(_batch_size, len(_X)//10)
    
    # 添加早停机制
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = _model.fit(
        _X, _y,
        epochs=_epochs,
        batch_size=adjusted_batch,
        validation_split=0.2,
        verbose=0,
        callbacks=[early_stop]
    )
    return history

# 主训练流程
def main_training():
    # 监控内存使用
    mem = psutil.virtual_memory()
    st.info(f"当前内存使用: {mem.used/(1024**3):.2f}GB / {mem.total/(1024**3):.2f}GB")
    if mem.available < 1 * 1024**3:  # <1GB时报警
        st.error("内存不足! 请减少时间步长或批大小")
        st.stop()
    
    # 创建数据集
    if 'X' not in st.session_state or 'y' not in st.session_state:
        X, y = create_dataset(scaled_data, timesteps)
        st.session_state.X = X
        st.session_state.y = y
    else:
        X = st.session_state.X
        y = st.session_state.y
        
    # 验证数据形状
    if len(X.shape) != 3:
        X = X.reshape(X.shape[0], X.shape[1], 1)
    st.write(f"输入数据形状: {X.shape} | 目标数据形状: {y.shape}")
    
    # 构建模型
    model = build_model(timesteps)
    
    # 训练模型
    st.subheader("模型训练过程")
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.info("模型训练中，请稍候...")
    
    start_time = time.time()
    try:
        history = train_model(model, X, y, epochs, batch_size)
    except Exception as e:
        import traceback
        st.error(f"训练失败: {str(e)}")
        st.code(traceback.format_exc())
        st.stop()
    
    training_time = time.time() - start_time
    progress_bar.progress(100)
    status_text.success(f"模型训练完成! 耗时: {training_time:.2f}秒")
    st.session_state['model'] = model
    
    # 绘制损失曲线
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history['loss'], label='训练损失')
    ax.plot(history.history['val_loss'], label='验证损失')
    ax.set_title('模型损失变化')
    ax.set_ylabel('损失')
    ax.set_xlabel('训练轮次')
    ax.legend()
    st.pyplot(fig)
    
    return model

# 检查是否已有训练好的模型
if 'model' not in st.session_state:
    model = main_training()
else:
    model = st.session_state['model']
    st.success("使用缓存的训练模型")

# ========== 预测与可视化 ==========
def predict_tomorrow():
    last_60_days = scaled_data[-timesteps:]
    last_60_days = last_60_days.reshape(1, timesteps, 1)
    
    with tf.device('/GPU:0'):
        pred_scaled = model.predict(last_60_days, verbose=0)
    
    return scaler.inverse_transform(pred_scaled)[0][0]

tomorrow_pred = predict_tomorrow()

# 计算历史准确性
if 'predictions' not in st.session_state:
    predictions = []
    actuals = hs300.iloc[timesteps:]['close'].values
    
    def predict_point(i):
        input_data = scaled_data[i - timesteps:i].reshape(1, timesteps, 1)
        pred = model.predict(input_data, verbose=0)
        return scaler.inverse_transform(pred)[0][0]
    
    # 并行预测
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(predict_point, i) 
                  for i in range(timesteps, len(scaled_data))]
        predictions = [f.result() for f in futures]
    
    st.session_state['predictions'] = predictions
    st.session_state['actuals'] = actuals
else:
    predictions = st.session_state['predictions']
    actuals = st.session_state['actuals']

errors = np.abs(predictions - actuals)
accuracy = 100 * (1 - errors / actuals)

# 显示结果
st.subheader("预测结果")
pred_date = next_trading_day.strftime("%Y-%m-%d")
change = tomorrow_pred - hs300['close'].iloc[-1]

st.metric(label=f"{pred_date} 预测收盘价",
          value=f"{tomorrow_pred:.2f}",
          delta=f"{change:.2f}",
          delta_color="normal")

# 可视化预测
st.subheader("近期表现与预测")
fig, ax = plt.subplots(figsize=(12, 6))

# 历史数据
ax.plot(hs300.index[-60:], hs300['close'].iloc[-60:], 'b-', label='历史数据')

# 预测点
prediction_point = last_date + pd.Timedelta(days=1)
ax.plot(prediction_point, tomorrow_pred, 'ro', markersize=8, label='预测值')

# 历史预测点
accuracy_df = pd.DataFrame({
    'date': hs300.index[timesteps:][-30:],
    'predicted': predictions[-30:]
})
ax.plot(accuracy_df['date'], accuracy_df['predicted'], 'g--', alpha=0.7, label='历史预测')

ax.set_title('沪深300指数预测')
ax.set_xlabel('日期')
ax.set_ylabel('收盘价')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

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

# 免责声明
st.divider()
st.caption("免责声明：本预测仅基于历史数据模型计算，不构成任何投资建议。市场有风险，投资需谨慎。")
