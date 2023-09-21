import numpy as np
import os
from utils import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_log_error
from scipy.stats import pearsonr
import time


class baseline_model:
    def __init__(self, train_x, train_y, test_x, test_y, data_max, data_min, root_path) -> None:
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.data_max = data_max
        self.data_min = data_min
        self.root_path = root_path

    def evaluate(self, model, y_true, pred, flag):
        # 保存评价指标结果
        mkdir(self.root_path + 'save/evaluation')
        with open(self.root_path + 'save/evaluation/' +  model + '_' + flag + '.txt', 'w') as f:
            f.write('MSE: ' + str(mean_squared_error(y_true.reshape(1,-1)[0], pred.reshape(1,-1)[0])) + '\n')
            f.write('RMSE: ' + str(np.sqrt(mean_squared_error(y_true.reshape(1,-1)[0], pred.reshape(1,-1)[0]))) + '\n')
            f.write('MAE: ' + str(mean_absolute_error(y_true.reshape(1,-1)[0], pred.reshape(1,-1)[0])) + '\n')
            f.write('MAPE: ' + str(mean_absolute_percentage_error(y_true.reshape(1,-1)[0], pred.reshape(1,-1)[0])) + '\n')
            f.write('R2: ' + str(r2_score(y_true.reshape(1,-1)[0], pred.reshape(1,-1)[0])) + '\n')
            f.write('Pearsonr: ' + str(pearsonr(y_true.reshape(1,-1)[0], pred.reshape(1,-1)[0])[0]) + '\n')

    # 线性回归
    def lr(self, flag):
        if os.path.exists(self.root_path + 'save/evaluation/lr_' + flag + '.txt'):
            print("lr has done!")
            return
        # 程序开始时间
        start_time = time.time()
        hyperparams = {
            'fit_intercept': [True, False],
        }
        lr = LinearRegression()
        grid = GridSearchCV(lr, hyperparams, cv=3, n_jobs=-1)
        print("lr is training...")
        if len(self.train_y.shape) == 3:
            for i in range(self.train_y.shape[2]):
                grid.fit(self.train_x[:,:,i], self.train_y[:,:,i].ravel())
        else:
            grid.fit(self.train_x, self.train_y.ravel())
        # 保存最好的模型
        mkdir(self.root_path + 'save/models')
        joblib.dump(grid.best_estimator_, self.root_path + 'save/models/lr_' + flag + '.pkl')
        print("lr is predicting...")
        # 预测
        if len(self.test_y.shape) == 3:
            num = 0
            for i in range(self.test_y.shape[2]):
                num += 1
                pr = grid.predict(self.test_x[:,:,i]).reshape(-1,1)
                if num == 1:
                    pred = pr
                else:
                    pred = np.concatenate((pred, pr), axis=1)
                # 保存预测结果图片
                mkdir(self.root_path + 'save/pics')
                save_result_pic(self.test_y[:,0,i], pr[:,0], self.root_path + 'save/pics/lr_' + flag + '_' + str(num) + '.png')
        else:
            pred = grid.predict(self.test_x)
            # 保存预测结果图片
            mkdir(self.root_path + 'save/pics')
            save_result_pic(self.test_y[:,0], pred, self.root_path + 'save/pics/lr_' + flag + '.png')
        # 反归一化
        pred = denormalize(pred, self.data_max, self.data_min)
        if len(self.test_y.shape) == 3:
            y_true = denormalize(self.test_y[:,0,:], self.data_max, self.data_min)
        else:
            y_true = denormalize(self.test_y, self.data_max, self.data_min)
        # 保存预测结果
        mkdir(self.root_path + 'save/results')
        np.savetxt(self.root_path + 'save/results/lr_' + flag + '.csv', pred, delimiter=',')
        # 保存评价指标结果
        self.evaluate('lr', y_true, pred, flag)
        print("lr is done!")
        # 程序结束时间
        end_time = time.time()
        print("lr execution time: ", int(end_time - start_time), "s")


    #  岭回归
    def ridge(self, flag):
        if os.path.exists(self.root_path + 'save/evaluation/ridge_' + flag + '.txt'):
            print("ridge has done!")
            return
        start_time = time.time()
        hyperparams = {
            'alpha': [0.01, 0.1, 1, 10, 100],
            'fit_intercept': [True, False],
        }
        ridge = Ridge()
        grid = RandomizedSearchCV(ridge, hyperparams, cv=3, n_jobs=-1)
        print("ridge is training...")
        if len(self.train_y.shape) == 3:
            for i in range(self.train_y.shape[2]):
                grid.fit(self.train_x[:,:,i], self.train_y[:,:,i].ravel())
        else:
            grid.fit(self.train_x, self.train_y.ravel())
        # 保存最好的模型
        mkdir(self.root_path + 'save/models')
        joblib.dump(grid.best_estimator_, self.root_path + 'save/models/ridge_' + flag + '.pkl')
        print("ridge is predicting...")
        # 预测
        if len(self.test_y.shape) == 3:
            num = 0
            for i in range(self.test_y.shape[2]):
                num += 1
                pr = grid.predict(self.test_x[:,:,i]).reshape(-1,1)
                if num == 1:
                    pred = pr
                else:
                    pred = np.concatenate((pred, pr), axis=1)
                # 保存预测结果图片
                mkdir(self.root_path + 'save/pics')
                save_result_pic(self.test_y[:,0,i], pr[:,0], self.root_path + 'save/pics/ridge_' + flag + '_' + str(num) + '.png')
        else:
            pred = grid.predict(self.test_x)
            # 保存预测结果图片
            mkdir(self.root_path + 'save/pics')
            save_result_pic(self.test_y[:,0], pred, self.root_path + 'save/pics/ridge_' + flag + '.png')
        # 反归一化
        pred = denormalize(pred, self.data_max, self.data_min)
        if len(self.test_y.shape) == 3:
            y_true = denormalize(self.test_y[:,0,:], self.data_max, self.data_min)
        else:
            y_true = denormalize(self.test_y, self.data_max, self.data_min)
        # 保存预测结果
        mkdir(self.root_path + 'save/results')
        np.savetxt(self.root_path + 'save/results/ridge_' + flag + '.csv', pred, delimiter=',')
        # 保存评价指标结果
        self.evaluate('ridge', y_true, pred, flag)
        print("ridge is done!")
        end_time = time.time()
        print("ridge execution time: ", int(end_time - start_time), "s")


    # lasso回归
    def lasso(self, flag):
        if os.path.exists(self.root_path + 'save/evaluation/lasso_' + flag + '.txt'):
            print("lasso has done!")
            return
        start_time = time.time()
        hyperparams = {
            'alpha': [0.01, 0.1, 1, 10, 100],
            'fit_intercept': [True, False],
        }
        lasso = Lasso()
        grid = RandomizedSearchCV(lasso, hyperparams, cv=3, n_jobs=-1)
        print("lasso is training...")
        if len(self.train_y.shape) == 3:
            for i in range(self.train_y.shape[2]):
                grid.fit(self.train_x[:,:,i], self.train_y[:,:,i].ravel())
        else:
            grid.fit(self.train_x, self.train_y.ravel())
        # 保存最好的模型
        mkdir(self.root_path + 'save/models')
        joblib.dump(grid.best_estimator_, self.root_path + 'save/models/lasso_' + flag + '.pkl')
        print("lasso is predicting...")
        # 预测
        if len(self.test_y.shape) == 3:
            num = 0
            for i in range(self.test_y.shape[2]):
                num += 1
                pr = grid.predict(self.test_x[:,:,i]).reshape(-1,1)
                if num == 1:
                    pred = pr
                else:
                    pred = np.concatenate((pred, pr), axis=1)
                # 保存预测结果图片
                mkdir(self.root_path + 'save/pics')
                save_result_pic(self.test_y[:,0,i], pr[:,0], self.root_path + 'save/pics/lasso_' + flag + '_' + str(num) + '.png')
        else:
            pred = grid.predict(self.test_x)
            # 保存预测结果图片
            mkdir(self.root_path + 'save/pics')
            save_result_pic(self.test_y[:,0], pred, self.root_path + 'save/pics/lasso_' + flag + '.png')
        # 反归一化
        pred = denormalize(pred, self.data_max, self.data_min)
        if len(self.test_y.shape) == 3:
            y_true = denormalize(self.test_y[:,0,:], self.data_max, self.data_min)
        else:
            y_true = denormalize(self.test_y, self.data_max, self.data_min)
        # 保存预测结果
        mkdir(self.root_path + 'save/results')
        np.savetxt(self.root_path + 'save/results/lasso_' + flag + '.csv', pred, delimiter=',')
        # 保存评价指标结果
        self.evaluate('lasso', y_true, pred, flag)
        print("lasso is done!")
        end_time = time.time()
        print("lasso execution time: ", int(end_time - start_time), "s")


    # 弹性网回归
    def elastic_net(self, flag):
        if os.path.exists(self.root_path + 'save/evaluation/elastic_net_' + flag + '.txt'):
            print("elastic_net has done!")
            return
        start_time = time.time()
        hyperparams = {
            'alpha': [0.01, 0.1, 1, 10, 100],
            'l1_ratio': [0.01, 0.1, 0.5, 0.7, 0.9],
            'fit_intercept': [True, False],
        }
        elastic_net = ElasticNet()
        grid = RandomizedSearchCV(elastic_net, hyperparams, cv=3, n_jobs=-1)
        print("elastic_net is training...")
        if len(self.train_y.shape) == 3:
            for i in range(self.train_y.shape[2]):
                grid.fit(self.train_x[:,:,i], self.train_y[:,:,i].ravel())
        else:
            grid.fit(self.train_x, self.train_y.ravel())
        # 保存最好的模型
        mkdir(self.root_path + 'save/models')
        joblib.dump(grid.best_estimator_, self.root_path + 'save/models/elastic_net_' + flag + '.pkl')
        print("elastic_net is predicting...")
        # 预测
        if len(self.test_y.shape) == 3:
            num = 0
            for i in range(self.test_y.shape[2]):
                num += 1
                pr = grid.predict(self.test_x[:,:,i]).reshape(-1,1)
                if num == 1:
                    pred = pr
                else:
                    pred = np.concatenate((pred, pr), axis=1)
                # 保存预测结果图片
                mkdir(self.root_path + 'save/pics')
                save_result_pic(self.test_y[:,0,i], pr[:,0], self.root_path + 'save/pics/elastic_net_' + flag + '_' + str(num) + '.png')
        else:
            pred = grid.predict(self.test_x)
            # 保存预测结果图片
            mkdir(self.root_path + 'save/pics')
            save_result_pic(self.test_y[:,0], pred, self.root_path + 'save/pics/elastic_net_' + flag + '.png')
        # 反归一化
        pred = denormalize(pred, self.data_max, self.data_min)
        if len(self.test_y.shape) == 3:
            y_true = denormalize(self.test_y[:,0,:], self.data_max, self.data_min)
        else:
            y_true = denormalize(self.test_y, self.data_max, self.data_min)
        # 保存预测结果
        mkdir(self.root_path + 'save/results')
        np.savetxt(self.root_path + 'save/results/elastic_net_' + flag + '.csv', pred, delimiter=',')
        # 保存评价指标结果
        self.evaluate('elastic_net', y_true, pred, flag)
        print("elastic_net is done!")
        end_time = time.time()
        print("elastic_net execution time: ", int(end_time - start_time), "s")


    # 贝叶斯岭回归
    def bayesian_ridge(self, flag):
        if os.path.exists(self.root_path + 'save/evaluation/bayesian_ridge_' + flag + '.txt'):
            print("bayesian_ridge has done!")
            return
        start_time = time.time()
        hyperparams = {
            'alpha_1': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            'alpha_2': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            'lambda_1': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            'lambda_2': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        }
        bayesian_ridge = BayesianRidge()
        grid = RandomizedSearchCV(bayesian_ridge, hyperparams, cv=3, n_jobs=-1)
        print("bayesian_ridge is training...")
        if len(self.train_y.shape) == 3:
            for i in range(self.train_y.shape[2]):
                grid.fit(self.train_x[:,:,i], self.train_y[:,:,i].ravel())
        else:
            grid.fit(self.train_x, self.train_y.ravel())
        # 保存最好的模型
        mkdir(self.root_path + 'save/models')
        joblib.dump(grid.best_estimator_, self.root_path + 'save/models/bayesian_ridge_' + flag + '.pkl')
        print("bayesian_ridge is predicting...")
        # 预测
        if len(self.test_y.shape) == 3:
            num = 0
            for i in range(self.test_y.shape[2]):
                num += 1
                pr = grid.predict(self.test_x[:,:,i]).reshape(-1,1)
                if num == 1:
                    pred = pr
                else:
                    pred = np.concatenate((pred, pr), axis=1)
                # 保存预测结果图片
                mkdir(self.root_path + 'save/pics')
                save_result_pic(self.test_y[:,0,i], pr[:,0], self.root_path + 'save/pics/bayesian_ridge_' + flag + '_' + str(num) + '.png')
        else:
            pred = grid.predict(self.test_x)
            # 保存预测结果图片
            mkdir(self.root_path + 'save/pics')
            save_result_pic(self.test_y[:,0], pred, self.root_path + 'save/pics/bayesian_ridge_' + flag + '.png')
        # 反归一化
        pred = denormalize(pred, self.data_max, self.data_min)
        if len(self.test_y.shape) == 3:
            y_true = denormalize(self.test_y[:,0,:], self.data_max, self.data_min)
        else:
            y_true = denormalize(self.test_y, self.data_max, self.data_min)
        # 保存预测结果
        mkdir(self.root_path + 'save/results')
        np.savetxt(self.root_path + 'save/results/bayesian_ridge_' + flag + '.csv', pred, delimiter=',')
        # 保存评价指标结果
        self.evaluate('bayesian_ridge', y_true, pred, flag)
        print("bayesian_ridge is done!")
        end_time = time.time()
        print("bayesian_ridge execution time: ", int(end_time - start_time), "s")


    # 支持向量机回归
    def svr(self, flag):
        if os.path.exists(self.root_path + 'save/evaluation/svr_' + flag + '.txt'):
            print("svr has done!")
            return
        start_time = time.time()
        hyperparams = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.01, 0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
        }
        svr = SVR()
        grid = RandomizedSearchCV(svr, hyperparams, cv=3, n_jobs=-1)
        print("svr is training...")
        if len(self.train_y.shape) == 3:
            for i in range(self.train_y.shape[2]):
                grid.fit(self.train_x[:,:,i], self.train_y[:,:,i].ravel())
        else:
            grid.fit(self.train_x, self.train_y.ravel())
        # 保存最好的模型
        mkdir(self.root_path + 'save/models')
        joblib.dump(grid.best_estimator_, self.root_path + 'save/models/svr_' + flag + '.pkl')
        print("svr is predicting...")
        # 预测
        if len(self.test_y.shape) == 3:
            num = 0
            for i in range(self.test_y.shape[2]):
                num += 1
                pr = grid.predict(self.test_x[:,:,i]).reshape(-1,1)
                if num == 1:
                    pred = pr
                else:
                    pred = np.concatenate((pred, pr), axis=1)
                # 保存预测结果图片
                mkdir(self.root_path + 'save/pics')
                save_result_pic(self.test_y[:,0,i], pr[:,0], self.root_path + 'save/pics/svr_' + flag + '_' + str(num) + '.png')
        else:
            pred = grid.predict(self.test_x)
            # 保存预测结果图片
            mkdir(self.root_path + 'save/pics')
            save_result_pic(self.test_y[:,0], pred, self.root_path + 'save/pics/svr_' + flag + '.png')
        # 反归一化
        pred = denormalize(pred, self.data_max, self.data_min)
        if len(self.test_y.shape) == 3:
            y_true = denormalize(self.test_y[:,0,:], self.data_max, self.data_min)
        else:
            y_true = denormalize(self.test_y, self.data_max, self.data_min)
        # 保存预测结果
        mkdir(self.root_path + 'save/results')
        np.savetxt(self.root_path + 'save/results/svr_' + flag + '.csv', pred, delimiter=',')
        # 保存评价指标结果
        self.evaluate('svr', y_true, pred, flag)
        print("svr is done!")
        end_time = time.time()
        print("svr execution time: ", int(end_time - start_time), "s")


    # K近邻回归
    def knn(self, flag):
        if os.path.exists(self.root_path + 'save/evaluation/knn_' + flag + '.txt'):
            print("knn has done!")
            return
        start_time = time.time()
        hyperparams = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        }
        knn = KNeighborsRegressor()
        grid = RandomizedSearchCV(knn, hyperparams, cv=3, n_jobs=-1)
        print("knn is training...")
        if len(self.train_y.shape) == 3:
            for i in range(self.train_y.shape[2]):
                grid.fit(self.train_x[:,:,i], self.train_y[:,:,i].ravel())
        else:
            grid.fit(self.train_x, self.train_y.ravel())
        # 保存最好的模型
        mkdir(self.root_path + 'save/models')
        joblib.dump(grid.best_estimator_, self.root_path + 'save/models/knn_' + flag + '.pkl')
        print("knn is predicting...")
        # 预测
        if len(self.test_y.shape) == 3:
            num = 0
            for i in range(self.test_y.shape[2]):
                num += 1
                pr = grid.predict(self.test_x[:,:,i]).reshape(-1,1)
                if num == 1:
                    pred = pr
                else:
                    pred = np.concatenate((pred, pr), axis=1)
                # 保存预测结果图片
                mkdir(self.root_path + 'save/pics')
                save_result_pic(self.test_y[:,0,i], pr[:,0], self.root_path + 'save/pics/knn_' + flag + '_' + str(num) + '.png')
        else:
            pred = grid.predict(self.test_x)
            # 保存预测结果图片
            mkdir(self.root_path + 'save/pics')
            save_result_pic(self.test_y[:,0], pred, self.root_path + 'save/pics/knn_' + flag + '.png')
        # 反归一化
        pred = denormalize(pred, self.data_max, self.data_min)
        if len(self.test_y.shape) == 3:
            y_true = denormalize(self.test_y[:,0,:], self.data_max, self.data_min)
        else:
            y_true = denormalize(self.test_y, self.data_max, self.data_min)
        # 保存预测结果
        mkdir(self.root_path + 'save/results')
        np.savetxt(self.root_path + 'save/results/knn_' + flag + '.csv', pred, delimiter=',')
        # 保存评价指标结果
        self.evaluate('knn', y_true, pred, flag)
        print("knn is done!")
        end_time = time.time()
        print("knn execution time: ", int(end_time - start_time), "s")

    
    # 决策树回归
    def dt(self, flag):
        if os.path.exists(self.root_path + 'save/evaluation/dt_' + flag + '.txt'):
            print("dt has done!")
            return
        start_time = time.time()
        hyperparams = {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 5, 10, 20, 50],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 3, 4, 5],
        }
        dt = DecisionTreeRegressor()
        grid = RandomizedSearchCV(dt, hyperparams, cv=3, n_jobs=-1)
        print("dt is training...")
        if len(self.train_y.shape) == 3:
            for i in range(self.train_y.shape[2]):
                grid.fit(self.train_x[:,:,i], self.train_y[:,:,i].ravel())
        else:
            grid.fit(self.train_x, self.train_y.ravel())
        # 保存最好的模型
        mkdir(self.root_path + 'save/models')
        joblib.dump(grid.best_estimator_, self.root_path + 'save/models/dt_' + flag + '.pkl')
        print("dt is predicting...")
        # 预测
        if len(self.test_y.shape) == 3:
            num = 0
            for i in range(self.test_y.shape[2]):
                num += 1
                pr = grid.predict(self.test_x[:,:,i]).reshape(-1,1)
                if num == 1:
                    pred = pr
                else:
                    pred = np.concatenate((pred, pr), axis=1)
                # 保存预测结果图片
                mkdir(self.root_path + 'save/pics')
                save_result_pic(self.test_y[:,0,i], pr[:,0], self.root_path + 'save/pics/dt_' + flag + '_' + str(num) + '.png')
        else:
            pred = grid.predict(self.test_x)
            # 保存预测结果图片
            mkdir(self.root_path + 'save/pics')
            save_result_pic(self.test_y[:,0], pred, self.root_path + 'save/pics/dt_' + flag + '.png')
        # 反归一化
        pred = denormalize(pred, self.data_max, self.data_min)
        if len(self.test_y.shape) == 3:
            y_true = denormalize(self.test_y[:,0,:], self.data_max, self.data_min)
        else:
            y_true = denormalize(self.test_y, self.data_max, self.data_min)
        # 保存预测结果
        mkdir(self.root_path + 'save/results')
        np.savetxt(self.root_path + 'save/results/dt_' + flag + '.csv', pred, delimiter=',')
        # 保存评价指标结果
        self.evaluate('dt', y_true, pred, flag)
        print("dt is done!")
        end_time = time.time()
        print("dt execution time: ", int(end_time - start_time), "s")


    # 随机森林回归
    def rf(self, flag):
        if os.path.exists(self.root_path + 'save/evaluation/rf_' + flag + '.txt'):
            print("rf has done!")
            return
        start_time = time.time()
        hyperparams = {
            'n_estimators': [50, 100, 200, 300],
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'max_depth': [None, 5, 10, 20, 50],
        }
        rf = RandomForestRegressor()
        grid = RandomizedSearchCV(rf, hyperparams, cv=3, n_jobs=-1)
        print("rf is training...")
        if len(self.train_y.shape) == 3:
            for i in range(self.train_y.shape[2]):
                grid.fit(self.train_x[:,:,i], self.train_y[:,:,i].ravel())
        else:
            grid.fit(self.train_x, self.train_y.ravel())
        # 保存最好的模型
        mkdir(self.root_path + 'save/models')
        joblib.dump(grid.best_estimator_, self.root_path + 'save/models/rf_' + flag + '.pkl')
        print("rf is predicting...")
        # 预测
        if len(self.test_y.shape) == 3:
            num = 0
            for i in range(self.test_y.shape[2]):
                num += 1
                pr = grid.predict(self.test_x[:,:,i]).reshape(-1,1)
                if num == 1:
                    pred = pr
                else:
                    pred = np.concatenate((pred, pr), axis=1)
                # 保存预测结果图片
                mkdir(self.root_path + 'save/pics')
                save_result_pic(self.test_y[:,0,i], pr[:,0], self.root_path + 'save/pics/rf_' + flag + '_' + str(num) + '.png')
        else:
            pred = grid.predict(self.test_x)
            # 保存预测结果图片
            mkdir(self.root_path + 'save/pics')
            save_result_pic(self.test_y[:,0], pred, self.root_path + 'save/pics/rf_' + flag + '.png')
        # 反归一化
        pred = denormalize(pred, self.data_max, self.data_min)
        if len(self.test_y.shape) == 3:
            y_true = denormalize(self.test_y[:,0,:], self.data_max, self.data_min)
        else:
            y_true = denormalize(self.test_y, self.data_max, self.data_min)
        # 保存预测结果
        mkdir(self.root_path + 'save/results')
        np.savetxt(self.root_path + 'save/results/rf_' + flag + '.csv', pred, delimiter=',')
        # 保存评价指标结果
        self.evaluate('rf', y_true, pred, flag)
        print("rf is done!")
        end_time = time.time()
        print("rf execution time: ", int(end_time - start_time), "s")


    # 梯度提升回归
    def gbdt(self, flag):
        if os.path.exists(self.root_path + 'save/evaluation/gbdt_' + flag + '.txt'):
            print("gbdt has done!")
            return
        start_time = time.time()
        hyperparams = {
            'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
            'learning_rate': [0.01, 0.1, 0.5, 1],
            'n_estimators': [50, 100, 200, 300],
            'criterion': ['friedman_mse', 'squared_error'],
            'max_depth': [None, 3, 5, 7],
        }
        gbdt = GradientBoostingRegressor()
        grid = RandomizedSearchCV(gbdt, hyperparams, cv=3, n_jobs=-1)
        print("gbdt is training...")
        if len(self.train_y.shape) == 3:
            for i in range(self.train_y.shape[2]):
                grid.fit(self.train_x[:,:,i], self.train_y[:,:,i].ravel())
        else:
            grid.fit(self.train_x, self.train_y.ravel())
        # 保存最好的模型
        mkdir(self.root_path + 'save/models')
        joblib.dump(grid.best_estimator_, self.root_path + 'save/models/gbdt_' + flag + '.pkl')
        print("gbdt is predicting...")
        # 预测
        if len(self.test_y.shape) == 3:
            num = 0
            for i in range(self.test_y.shape[2]):
                num += 1
                pr = grid.predict(self.test_x[:,:,i]).reshape(-1,1)
                if num == 1:
                    pred = pr
                else:
                    pred = np.concatenate((pred, pr), axis=1)
                # 保存预测结果图片
                mkdir(self.root_path + 'save/pics')
                save_result_pic(self.test_y[:,0,i], pr[:,0], self.root_path + 'save/pics/gbdt_' + flag + '_' + str(num) + '.png')
        else:
            pred = grid.predict(self.test_x)
            # 保存预测结果图片
            mkdir(self.root_path + 'save/pics')
            save_result_pic(self.test_y[:,0], pred, self.root_path + 'save/pics/gbdt_' + flag + '.png')
        # 反归一化
        pred = denormalize(pred, self.data_max, self.data_min)
        if len(self.test_y.shape) == 3:
            y_true = denormalize(self.test_y[:,0,:], self.data_max, self.data_min)
        else:
            y_true = denormalize(self.test_y, self.data_max, self.data_min)
        # 保存预测结果
        mkdir(self.root_path + 'save/results')
        np.savetxt(self.root_path + 'save/results/gbdt_' + flag + '.csv', pred, delimiter=',')
        # 保存评价指标结果
        self.evaluate('gbdt', y_true, pred, flag)
        print("gbdt is done!")
        end_time = time.time()
        print("gbdt execution time: ", int(end_time - start_time), "s")


    # AdaBoost回归
    def adaboost(self, flag):
        if os.path.exists(self.root_path + 'save/evaluation/adaboost_' + flag + '.txt'):
            print("adaboost has done!")
            return
        start_time = time.time()
        hyperparams = {
            'loss': ['linear', 'square', 'exponential'],
            'learning_rate': [0.01, 0.1, 0.5, 1],
            'n_estimators': [50, 100, 200, 300],
        }
        adaboost = AdaBoostRegressor()
        grid = RandomizedSearchCV(adaboost, hyperparams, cv=3, n_jobs=-1)
        print("adaboost is training...")
        if len(self.train_y.shape) == 3:
            for i in range(self.train_y.shape[2]):
                grid.fit(self.train_x[:,:,i], self.train_y[:,:,i].ravel())
        else:
            grid.fit(self.train_x, self.train_y.ravel())
        # 保存最好的模型
        mkdir(self.root_path + 'save/models')
        joblib.dump(grid.best_estimator_, self.root_path + 'save/models/adaboost_' + flag + '.pkl')
        print("adaboost is predicting...")
        # 预测
        if len(self.test_y.shape) == 3:
            num = 0
            for i in range(self.test_y.shape[2]):
                num += 1
                pr = grid.predict(self.test_x[:,:,i]).reshape(-1,1)
                if num == 1:
                    pred = pr
                else:
                    pred = np.concatenate((pred, pr), axis=1)
                # 保存预测结果图片
                mkdir(self.root_path + 'save/pics')
                save_result_pic(self.test_y[:,0,i], pr[:,0], self.root_path + 'save/pics/adaboost_' + flag + '_' + str(num) + '.png')
        else:
            pred = grid.predict(self.test_x)
            # 保存预测结果图片
            mkdir(self.root_path + 'save/pics')
            save_result_pic(self.test_y[:,0], pred, self.root_path + 'save/pics/adaboost_' + flag + '.png')
        # 反归一化
        pred = denormalize(pred, self.data_max, self.data_min)
        if len(self.test_y.shape) == 3:
            y_true = denormalize(self.test_y[:,0,:], self.data_max, self.data_min)
        else:
            y_true = denormalize(self.test_y, self.data_max, self.data_min)
        # 保存预测结果
        mkdir(self.root_path + 'save/results')
        np.savetxt(self.root_path + 'save/results/adaboost_' + flag + '.csv', pred, delimiter=',')
        # 保存评价指标结果
        self.evaluate('adaboost', y_true, pred, flag)
        print("adaboost is done!")
        end_time = time.time()
        print("adaboost execution time: ", int(end_time - start_time), "s")

    
    # XGBoost回归
    def xgb(self, flag):
        if os.path.exists(self.root_path + 'save/evaluation/xgb_' + flag + '.txt'):
            print("xgb has done!")
            return
        start_time = time.time()
        hyperparams = {
            'learning_rate': [0.01, 0.1, 0.5, 1],
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'gamma': [0.1, 0.5, 0.9],
            'reg_lambda': [0.01, 0.1, 1, 10],
            'reg_alpha': [0.01, 0.1, 1, 10],
        }
        xgb = XGBRegressor()
        grid = RandomizedSearchCV(xgb, hyperparams, cv=3, n_jobs=-1)
        print("xgb is training...")
        if len(self.train_y.shape) == 3:
            for i in range(self.train_y.shape[2]):
                grid.fit(self.train_x[:,:,i], self.train_y[:,:,i].ravel())
        else:
            grid.fit(self.train_x, self.train_y.ravel())
        # 保存最好的模型
        mkdir(self.root_path + 'save/models')
        joblib.dump(grid.best_estimator_, self.root_path + 'save/models/xgb_' + flag + '.pkl')
        print("xgb is predicting...")
        # 预测
        if len(self.test_y.shape) == 3:
            num = 0
            for i in range(self.test_y.shape[2]):
                num += 1
                pr = grid.predict(self.test_x[:,:,i]).reshape(-1,1)
                if num == 1:
                    pred = pr
                else:
                    pred = np.concatenate((pred, pr), axis=1)
                # 保存预测结果图片
                mkdir(self.root_path + 'save/pics')
                save_result_pic(self.test_y[:,0,i], pr[:,0], self.root_path + 'save/pics/xgb_' + flag + '_' + str(num) + '.png')
        else:
            pred = grid.predict(self.test_x)
            # 保存预测结果图片
            mkdir(self.root_path + 'save/pics')
            save_result_pic(self.test_y[:,0], pred, self.root_path + 'save/pics/xgb_' + flag + '.png')
        # 反归一化
        pred = denormalize(pred, self.data_max, self.data_min)
        if len(self.test_y.shape) == 3:
            y_true = denormalize(self.test_y[:,0,:], self.data_max, self.data_min)
        else:
            y_true = denormalize(self.test_y, self.data_max, self.data_min)
        # 保存预测结果
        mkdir(self.root_path + 'save/results')
        np.savetxt(self.root_path + 'save/results/xgb_' + flag + '.csv', pred, delimiter=',')
        # 保存评价指标结果
        self.evaluate('xgb', y_true, pred, flag)
        print("xgb is done!")
        end_time = time.time()
        print("xgb execution time: ", int(end_time - start_time), "s")


    # LightGBM回归
    def lgb(self, flag):
        if os.path.exists(self.root_path + 'save/evaluation/lgb_' + flag + '.txt'):
            print("lgb has done!")
            return
        start_time = time.time()
        hyperparams = {
            'learning_rate': [0.01, 0.1, 0.5, 1],
            'n_estimators': [50, 100, 200, 300],
            'num_leaves': [31, 63, 127, 255],
            'reg_lambda': [0.01, 0.1, 1, 10],
            'reg_alpha': [0.01, 0.1, 1, 10],
        }
        lgb = LGBMRegressor()
        grid = RandomizedSearchCV(lgb, hyperparams, cv=3, n_jobs=-1)
        print("lgb is training...")
        if len(self.train_y.shape) == 3:
            for i in range(self.train_y.shape[2]):
                grid.fit(self.train_x[:,:,i], self.train_y[:,:,i].ravel())
        else:
            grid.fit(self.train_x, self.train_y.ravel())
        # 保存最好的模型
        mkdir(self.root_path + 'save/models')
        joblib.dump(grid.best_estimator_, self.root_path + 'save/models/lgb_' + flag + '.pkl')
        print("lgb is predicting...")
        # 预测
        if len(self.test_y.shape) == 3:
            num = 0
            for i in range(self.test_y.shape[2]):
                num += 1
                pr = grid.predict(self.test_x[:,:,i]).reshape(-1,1)
                if num == 1:
                    pred = pr
                else:
                    pred = np.concatenate((pred, pr), axis=1)
                # 保存预测结果图片
                mkdir(self.root_path + 'save/pics')
                save_result_pic(self.test_y[:,0,i], pr[:,0], self.root_path + 'save/pics/lgb_' + flag + '_' + str(num) + '.png')
        else:
            pred = grid.predict(self.test_x)
            # 保存预测结果图片
            mkdir(self.root_path + 'save/pics')
            save_result_pic(self.test_y[:,0], pred, self.root_path + 'save/pics/lgb_' + flag + '.png')
        # 反归一化
        pred = denormalize(pred, self.data_max, self.data_min)
        if len(self.test_y.shape) == 3:
            y_true = denormalize(self.test_y[:,0,:], self.data_max, self.data_min)
        else:
            y_true = denormalize(self.test_y, self.data_max, self.data_min)
        # 保存预测结果
        mkdir(self.root_path + 'save/results')
        np.savetxt(self.root_path + 'save/results/lgb_' + flag + '.csv', pred, delimiter=',')
        # 保存评价指标结果
        self.evaluate('lgb', y_true, pred, flag)
        print("lgb is done!")
        end_time = time.time()
        print("lgb execution time: ", int(end_time - start_time), "s")


    # CatBoost回归
    def cat(self, flag):
        if os.path.exists(self.root_path + 'save/evaluation/cat_' + flag + '.txt'):
            print("cat has done!")
            return
        start_time = time.time()
        hyperparams = {
            'learning_rate': [0.01, 0.1, 0.5, 1],
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'l2_leaf_reg': [1, 3, 5, 7],
        }
        cat = CatBoostRegressor()
        grid = RandomizedSearchCV(cat, hyperparams, cv=3, n_jobs=-1)
        print("cat is training...")
        if len(self.train_y.shape) == 3:
            for i in range(self.train_y.shape[2]):
                grid.fit(self.train_x[:,:,i], self.train_y[:,:,i].ravel(), verbose=False)
        else:
            grid.fit(self.train_x, self.train_y.ravel(), verbose=False)
        # 保存最好的模型
        mkdir(self.root_path + 'save/models')
        joblib.dump(grid.best_estimator_, self.root_path + 'save/models/cat_' + flag + '.pkl')
        print("cat is predicting...")
        # 预测
        if len(self.test_y.shape) == 3:
            num = 0
            for i in range(self.test_y.shape[2]):
                num += 1
                pr = grid.predict(self.test_x[:,:,i]).reshape(-1,1)
                if num == 1:
                    pred = pr
                else:
                    pred = np.concatenate((pred, pr), axis=1)
                # 保存预测结果图片
                mkdir(self.root_path + 'save/pics')
                save_result_pic(self.test_y[:,0,i], pr[:,0], self.root_path + 'save/pics/cat_' + flag + '_' + str(num) + '.png')
        else:
            pred = grid.predict(self.test_x)
            # 保存预测结果图片
            mkdir(self.root_path + 'save/pics')
            save_result_pic(self.test_y[:,0], pred, self.root_path + 'save/pics/cat_' + flag + '.png')
        # 反归一化
        pred = denormalize(pred, self.data_max, self.data_min)
        if len(self.test_y.shape) == 3:
            y_true = denormalize(self.test_y[:,0,:], self.data_max, self.data_min)
        else:
            y_true = denormalize(self.test_y, self.data_max, self.data_min)
        # 保存预测结果
        mkdir(self.root_path + 'save/results')
        np.savetxt(self.root_path + 'save/results/cat_' + flag + '.csv', pred, delimiter=',')
        # 保存评价指标结果
        self.evaluate('cat', y_true, pred, flag)
        print("cat is done!")
        end_time = time.time()
        print("cat execution time: ", int(end_time - start_time), "s")



    # 极端随机树回归
    def et(self, flag):
        if os.path.exists(self.root_path + 'save/evaluation/et_' + flag + '.txt'):
            print("et has done!")
            return
        start_time = time.time()
        hyperparams = {
            'n_estimators': [50, 100, 200, 300],
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'max_depth': [None, 5, 10, 20, 50],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 3, 4, 5],
        }
        et = ExtraTreesRegressor()
        grid = RandomizedSearchCV(et, hyperparams, cv=3, n_jobs=-1)
        print("et is training...")
        if len(self.train_y.shape) == 3:
            for i in range(self.train_y.shape[2]):
                grid.fit(self.train_x[:,:,i], self.train_y[:,:,i].ravel())
        else:
            grid.fit(self.train_x, self.train_y.ravel())
        # 保存最好的模型
        mkdir(self.root_path + 'save/models')
        joblib.dump(grid.best_estimator_, self.root_path + 'save/models/et_' + flag + '.pkl')
        print("et is predicting...")
        # 预测
        if len(self.test_y.shape) == 3:
            num = 0
            for i in range(self.test_y.shape[2]):
                num += 1
                pr = grid.predict(self.test_x[:,:,i]).reshape(-1,1)
                if num == 1:
                    pred = pr
                else:
                    pred = np.concatenate((pred, pr), axis=1)
                # 保存预测结果图片
                mkdir(self.root_path + 'save/pics')
                save_result_pic(self.test_y[:,0,i], pr[:,0], self.root_path + 'save/pics/et_' + flag + '_' + str(num) + '.png')
        else:
            pred = grid.predict(self.test_x)
            # 保存预测结果图片
            mkdir(self.root_path + 'save/pics')
            save_result_pic(self.test_y[:,0], pred, self.root_path + 'save/pics/et_' + flag + '.png')
        # 反归一化
        pred = denormalize(pred, self.data_max, self.data_min)
        if len(self.test_y.shape) == 3:
            y_true = denormalize(self.test_y[:,0,:], self.data_max, self.data_min)
        else:
            y_true = denormalize(self.test_y, self.data_max, self.data_min)
        # 保存预测结果
        mkdir(self.root_path + 'save/results')
        np.savetxt(self.root_path + 'save/results/et_' + flag + '.csv', pred, delimiter=',')
        # 保存评价指标结果
        self.evaluate('et', y_true, pred, flag)
        print("et is done!")
        end_time = time.time()
        print("et execution time: ", int(end_time - start_time), "s")


    # 多层感知机回归
    def mlp(self, flag):
        if os.path.exists(self.root_path + 'save/evaluation/mlp_' + flag + '.txt'):
            print("mlp has done!")
            return
        start_time = time.time()
        hyperparams = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100, 100)],
        }
        mlp = MLPRegressor()
        grid = GridSearchCV(mlp, hyperparams, cv=3, n_jobs=-1)
        print("mlp is training...")
        if len(self.train_y.shape) == 3:
            for i in range(self.train_y.shape[2]):
                grid.fit(self.train_x[:,:,i], self.train_y[:,:,i].ravel())
        else:
            grid.fit(self.train_x, self.train_y.ravel())
        # 保存最好的模型
        mkdir(self.root_path + 'save/models')
        joblib.dump(grid.best_estimator_, self.root_path + 'save/models/mlp_' + flag + '.pkl')
        print("mlp is predicting...")
        # 预测
        if len(self.test_y.shape) == 3:
            num = 0
            for i in range(self.test_y.shape[2]):
                num += 1
                pr = grid.predict(self.test_x[:,:,i]).reshape(-1,1)
                if num == 1:
                    pred = pr
                else:
                    pred = np.concatenate((pred, pr), axis=1)
                # 保存预测结果图片
                mkdir(self.root_path + 'save/pics')
                save_result_pic(self.test_y[:,0,i], pr[:,0], self.root_path + 'save/pics/mlp_' + flag + '_' + str(num) + '.png')
        else:
            pred = grid.predict(self.test_x)
            # 保存预测结果图片
            mkdir(self.root_path + 'save/pics')
            save_result_pic(self.test_y[:,0], pred, self.root_path + 'save/pics/mlp_' + flag + '.png')
        # 反归一化
        pred = denormalize(pred, self.data_max, self.data_min)
        if len(self.test_y.shape) == 3:
            y_true = denormalize(self.test_y[:,0,:], self.data_max, self.data_min)
        else:
            y_true = denormalize(self.test_y, self.data_max, self.data_min)
        # 保存预测结果
        mkdir(self.root_path + 'save/results')
        np.savetxt(self.root_path + 'save/results/mlp_' + flag + '.csv', pred, delimiter=',')
        # 保存评价指标结果
        self.evaluate('mlp', y_true, pred, flag)
        print("mlp is done!")
        end_time = time.time()
        print("mlp execution time: ", int(end_time - start_time), "s")


    # 高斯过程回归
    def gpr(self, flag):
        if os.path.exists(self.root_path + 'save/evaluation/gpr_' + flag + '.txt'):
            print("gpr has done!")
            return
        start_time = time.time()
        hyperparams = {
            'alpha': [1e-10, 1e-5, 1e-2],
        }
        gpr = GaussianProcessRegressor()
        grid = GridSearchCV(gpr, hyperparams, cv=3, n_jobs=-1)
        print("gpr is training...")
        if len(self.train_y.shape) == 3:
            for i in range(self.train_y.shape[2]):
                grid.fit(self.train_x[:,:,i], self.train_y[:,:,i].ravel())
        else:
            grid.fit(self.train_x, self.train_y.ravel())
        # 保存最好的模型
        mkdir(self.root_path + 'save/models')
        joblib.dump(grid.best_estimator_, self.root_path + 'save/models/gpr_' + flag + '.pkl')
        print("gpr is predicting...")
        # 预测
        if len(self.test_y.shape) == 3:
            num = 0
            for i in range(self.test_y.shape[2]):
                num += 1
                pr = grid.predict(self.test_x[:,:,i]).reshape(-1,1)
                if num == 1:
                    pred = pr
                else:
                    pred = np.concatenate((pred, pr), axis=1)
                # 保存预测结果图片
                mkdir(self.root_path + 'save/pics')
                save_result_pic(self.test_y[:,0,i], pr[:,0], self.root_path + 'save/pics/gpr_' + flag + '_' + str(num) + '.png')
        else:
            pred = grid.predict(self.test_x)
            # 保存预测结果图片
            mkdir(self.root_path + 'save/pics')
            save_result_pic(self.test_y[:,0], pred, self.root_path + 'save/pics/gpr_' + flag + '.png')
        # 反归一化
        pred = denormalize(pred, self.data_max, self.data_min)
        if len(self.test_y.shape) == 3:
            y_true = denormalize(self.test_y[:,0,:], self.data_max, self.data_min)
        else:
            y_true = denormalize(self.test_y, self.data_max, self.data_min)
        # 保存预测结果
        mkdir(self.root_path + 'save/results')
        np.savetxt(self.root_path + 'save/results/gpr_' + flag + '.csv', pred, delimiter=',')
        # 保存评价指标结果
        self.evaluate('gpr', y_true, pred, flag)
        print("gpr is done!")
        end_time = time.time()
        print("gpr execution time: ", int(end_time - start_time), "s")





    def run(self, model, flag):
        if model == 'lr':
            self.lr(flag)
        elif model == 'ridge':
            self.ridge(flag)
        elif model == 'lasso':
            self.lasso(flag)
        elif model == 'elastic_net':
            self.elastic_net(flag)
        elif model == 'bayesian_ridge':
            self.bayesian_ridge(flag)
        elif model == 'svr':
            self.svr(flag)
        elif model == 'knn':
            self.knn(flag)
        elif model == 'dt':
            self.dt(flag)
        elif model == 'rf':
            self.rf(flag)
        elif model == 'gbdt':
            self.gbdt(flag)
        elif model == 'adaboost':
            self.adaboost(flag)
        elif model == 'xgb':
            self.xgb(flag)
        elif model == 'lgb':
            self.lgb(flag)
        elif model == 'cat':
            self.cat(flag)
        elif model == 'et':
            self.et(flag)
        elif model == 'mlp':
            self.mlp(flag)
        elif model == 'gpr':
            self.gpr(flag)
        else:
            pass
