from itertools import islice
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import torch
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.pandas import PandasDataset
import pandas as pd
import numpy as np
import sys, os

PROJ_DIR = "E:\Study\Lab\TimeSeriesAD\lag-llama-tuning\lag-llama"
sys.path.append(os.path.join(PROJ_DIR))
from lag_llama.gluon.estimator import LagLlamaEstimator


def get_lag_llama_predictions(dataset, prediction_length, num_samples=100):
    ckpt = torch.load(
        "content/lag-llama.ckpt", map_location=torch.device("cuda:0")
    )  # Uses GPU since in this Colab we use a GPU.
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    estimator = LagLlamaEstimator(
        ckpt_path="content/lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=32,  # Should not be changed; this is what the released Lag-Llama model was trained with
        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        batch_size=1,
        num_parallel_samples=100,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset, predictor=predictor, num_samples=num_samples
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    return forecasts, tss


def load_data(path, target_col):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # df.plot()

    # plt.savefig('lab/result/init.png')
    # print(df)

    # 将数值列设置为 float32 格式
    for col in df.columns:
        # 检查列的类型不是字符串类型
        if df[col].dtype != "object" and pd.api.types.is_string_dtype(df[col]) == False:
            df[col] = df[col].astype("float32")

    # 从长格式数据框创建 PandasDataset
    dataset = PandasDataset(df, target=target_col)
    return dataset


def test_passengers():
    dataset = load_data("lab/dataset/AirPassengers.csv", "Passengers")
    backtest_dataset = dataset
    prediction_length = 24  # 定义预测长度
    num_samples = 50  # 每个时间步从概率分布中抽取的样本数
    forecasts, tss = get_lag_llama_predictions(
        backtest_dataset, prediction_length, num_samples
    )

    # plt.figure(figsize=(20, 15))  # 设置绘图区域的大小为 20x15
    date_formater = mdates.DateFormatter("%Y-%m")
    # plt.rcParams.update({'font.size': 20})  # 更新字体大小为 15

    # 遍历前 9 个时间序列，并绘制预测样本
    for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
        # plt.plot(ts[-4 * prediction_length:].to_timestamp(), label="target")  # 绘制实际数据
        plt.plot(ts.to_timestamp(), label="target")
        print(type(forecast))
        forecast.plot(color="r", name="predict")  # 绘制预测结果
        plt.xticks(rotation=60)  # x 轴标签旋转 60 度

    plt.gcf().tight_layout()  # 调整子图布局使其紧凑显示
    plt.legend()  # 显示图例
    # plt.show()  # 展示绘制的图形
    plt.savefig("lab/result/predict_passengers.png")


def create_inout_sequences(input_data, time_window, predict_len):
    # 创建时间序列数据专用的数据分割器
    inout_seq = []
    L = len(input_data)
    for i in range(L - time_window):
        train_seq = input_data[i : i + time_window]
        if (i + time_window + predict_len) > len(input_data):
            break
        train_label = input_data[i + time_window : i + time_window + predict_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def window_splitter(input_data, lag_window, predict_window):
    # 评估时滑动步长为predict_window
    step = predict_window
    output_seq = []
    L = len(input_data)
    for i in range((L - lag_window) / step):
        use_seq = input_data[i : i + lag_window]
        if (i * step + lag_window + step) > L:
            break
        output_seq.append(use_seq)
    return output_seq


def test_swat():
    path = "lab/dataset/swat/SWaT_test.csv"
    df_origin = pd.read_csv(
        path, usecols=[0, 2, -1], index_col=0, parse_dates=True, dayfirst=True
    )
    # pandas.core.frame.DataFrame
    max_context_length_of_laglallma = 1092
    predict_length = 6
    context_length = predict_length*3
    df_list = window_splitter(df_origin, max_context_length_of_laglallma, predict_length)



if __name__ == "__main__":
    # test_swat()
    test_passengers()
