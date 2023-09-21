# Machine Learning Time Series Forecasting
用于时间序列预测的主流机器学习方法工具包.

### Method

- [x] Linear Regression
- [x] Ridge Regression
- [x] Lasso Regression
- [x] Elastic Net
- [x] Bayesian Ridge Regression
- [x] Support Vector Regression
- [x] K-Nearest Neighbors
- [x] Decision Tree
- [x] Random Forest
- [x] Gradient Boosting Regression
- [x] AdaBoost
- [x] XGBoost
- [x] LightGBM
- [x] CatBoost
- [x] Extra Trees Regression
- [x] Multi-layer Perceptron
- [x] Gaussian Process Regression

### Input

**单变量预测（单特征或多特征）：**

`train_x.shape #[samples, features] `

`train_y.shape #[samples, target]`

**多变量预测（多输入多输出，单特征或多特征）：**

这种情况相当于多条序列并行预测.

`train_x.shape #[samples, features, seq_nums]`

`train_y.shape #[samples, target, seq_nums]`

### Output

```shell
save
├─ results # 预测结果
│  ├─ xxx_COVID19_US_daily_case.csv
├─ pics # 预测效果保存目录
│  ├─ xxx_COVID19_US_daily_case_1.png
├─ models # 最优模型保存目录
│  ├─ xxx_COVID19_US_daily_case.pkl
└─ evaluation # 评估指标结果目录
   ├─ xxx_COVID19_US_daily_case.txt
```

### Quick Start

```shell
conda create -n mltool python=3.8
conda activate mltool
python -m pip install -r requirements.txt
```

欢迎批评指正. E-mail: jenqyanghou@gamil.com
