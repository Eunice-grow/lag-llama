import pandas as pd  # 导入pandas库，用于数据处理和分析
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于数据可视化
import matplotlib.dates as mdates  # 导入matplotlib的dates模块，用于处理日期格式
import torch  # 导入PyTorch库，用于深度学习模型的开发和训练

from itertools import islice  # 从itertools库导入islice函数，用于对迭代器进行切片操作

# 导入GluonTS库中的相关模块，用于时间序列预测
from gluonts.evaluation import make_evaluation_predictions, Evaluator  
from gluonts.dataset.repository.datasets import get_dataset  
import sys,os
PROJ_DIR = 'E:\Study\Lab\cloud-robot\study\lag-llama-tuning\lag-llama'
sys.path.append(os.path.join(PROJ_DIR))
from lag_llama.gluon.estimator import LagLlamaEstimator  

# 直接从GluonTS加载数据集
dataset = get_dataset("australian_electricity_demand")  # 加载澳大利亚电力需求数据集
backtest_dataset = dataset.test  # 获取测试数据集
prediction_length = dataset.metadata.prediction_length  # 获取预测长度
context_length = 3 * prediction_length  # 设置上下文长度为预测长度的3倍

# 预测
# 简单地初始化模型并使用LagLlamaEstimator对象
ckpt = torch.load("./content/lag-llama.ckpt", map_location=torch.device('cuda:0'))  # 加载预训练的模型权重
estimator_args = ckpt["hyper_parameters"]["model_kwargs"]  # 获取模型的超参数
# 初始化LagLlamaEstimator对象
estimator = LagLlamaEstimator(ckpt_path="./content/lag-llama.ckpt",
                              prediction_length=prediction_length,
                              context_length=context_length,
                              input_size=estimator_args["input_size"],
                              n_layer=estimator_args["n_layer"],
                              n_embd_per_head=estimator_args["n_embd_per_head"],
                              n_head=estimator_args["n_head"],
                              scaling=estimator_args["scaling"],
                              time_feat=estimator_args["time_feat"])

lightning_module = estimator.create_lightning_module()  # 创建Lightning模块
transformation = estimator.create_transformation()  # 创建数据预处理转换
predictor = estimator.create_predictor(transformation, lightning_module)  # 创建预测器

# 使用make_evaluation_predictions函数生成零样本的预测
forecast_it, ts_it = make_evaluation_predictions(dataset=backtest_dataset, predictor=predictor)

# 这个函数返回生成器。我们需要把它们转换成列表。
forecasts = list(forecast_it)  # 将预测结果转换为列表
tss = list(ts_it)  # 将真实时间序列转换为列表

# 评估
# GluonTS可以使用Evaluator对象方便地计算不同的性能指标。
evaluator = Evaluator() 
agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts))  # 计算评估指标

# 我们还可以随意地将预测可视化。
plt.figure(figsize=(20, 15))  # 设置图表大小
date_formater = mdates.DateFormatter('%b, %d')  # 设置日期格式
plt.rcParams.update({'font.size': 15})  # 更新字体大小

# 遍历前4个时间序列和预测结果，进行可视化
for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 4):
    ax = plt.subplot(2, 2, idx+1)  # 创建子图
    plt.plot(ts[-4 * dataset.metadata.prediction_length:].to_timestamp(), label="target")  # 绘制真实时间序列
    forecast.plot(color='g')  # 绘制预测结果

    plt.xticks(rotation=60)  # 旋转x轴标签
    ax.xaxis.set_major_formatter(date_formater)  # 设置x轴日期格式
    ax.set_title(forecast.item_id)  # 设置子图标题

plt.gcf().tight_layout()  # 调整整个图表的布局
plt.legend()  # 显示图例
plt.show()  # 显示图表