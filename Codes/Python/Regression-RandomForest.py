import pandas as pd
import numpy as np
import os
import xlsxwriter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score#R 2
from sklearn.metrics import mean_squared_error #MSE
from sklearn.metrics import mean_absolute_error #MAE

#训练模型，并预测，返回RMSE、R_Squared
def randomforenst_regressor(features,labels):
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,random_state=42)
    # 实例化、训练模型
    # Instantiate model
    rf = RandomForestRegressor(bootstrap=True,
                               max_depth= 40,
                               max_features=3,
                               min_samples_leaf= 3,
                               min_samples_split= 7,
                               n_estimators=350, random_state=42)
    # Train the model on training data
    rf.fit(train_features, train_labels)
    # 测试模型、打分
    predictions = rf.predict(test_features)
    #print(np.mean(predictions))
    RMSE=np.sqrt(mean_squared_error(test_labels, predictions))
    R2=r2_score(test_labels,predictions)
    errors = abs(predictions - test_labels)
    #mape = 100 * (errors / test_labels)
    #accuracy = 100 - mape
    print('Root Mean Squared Error:',RMSE)#误差越大，该值越大
    print('R_Squared:',R2)#衡量标准就是正确率，而正确率又在0～1之间，最高百分之百，最低0
    #print('Mean Absolute Error:',mean_absolute_error(test_labels, predictions))#误差越大，该值越大
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')#MAE，全称是Mean Absolute Error，即平均绝对值误差，它表示预测值和观测值之间绝对误差的平均值
    #print('Accuracy:', round(accuracy, 2), '%.')
    # 特征重要性
    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    return importances,predictions,train_features, test_features, train_labels, test_labels,RMSE,R2

#特征重要性可视化并输出
def importance_Visualized(importances,feature_list,outpathfigure,filename):
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    # list of x locations for plotting
    x_values = list(range(len(importances)))
    print(x_values)
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    #plt.figure(figsize=(15,8))
    #mpl.rcParams['font.sans-serif'] = ['Times New Roman']
    #mpl.rcParams['font.weight'] = 'bold'
    #mpl.rcParams['font.size'] = 26
    #plt.subplot(121)
    #设置字体
    plt.rcParams['font.family'] = ['Times New Roman']
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5), dpi=200)
    labels = feature_list
    x = np.arange(len(labels))  # the label locations
    #设置x，y标签大小
    label_font = {
        'weight': 'bold',
        'size': 8,
        'family': 'Times New Roman'
    }
    #设置标题大小
    font={
        'weight': 'bold',
        'size': 10,
        'family': 'Times New Roman'
    }
    # Set the style
    #plt.style.use('fivethirtyeight')
    #设置颜色和柱子宽度
    plt.bar(x_values, importances, color="Chocolate", width=0.4,orientation='vertical')
    # Make a bar chart
    #plt.bar(x_values, importances, orientation='vertical')
    # Tick labels for x axis
    plt.xticks(x_values, feature_list, rotation='vertical')
    # Axis labels and title
    plt.ylabel('Importance',label_font);
    plt.xlabel('Variable',label_font);
    plt.title('Variable Importances',font)
    # 设置坐标轴选旋转labelrotation=15,及坐标轴大小
    ax.tick_params(axis='x', labelsize=6, bottom=False, labelrotation=15)
    ax.tick_params(axis='y', labelsize=6, bottom=False)
    # 自定义x坐标轴标签
    ax.set_xticks(x)
    #设置网格
    plt.grid(True)
    # 动调整子图参数，使之填充整个图像区域
    plt.tight_layout(pad=2)
    #导出图像
    plt.savefig(outpathfigure+filename+'.png',dpi=600,bbox_inches = 'tight')

#调参：RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
def randomizeSearchCV(train_features,train_labels):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=300, stop=800, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(50, 100, num=10)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [7,8,9,10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [3,4,5,6]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    # Import the model we are using
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   n_iter=100, scoring='neg_root_mean_squared_error',
                                   cv=3, verbose=2, random_state=42, n_jobs=-1)

    # Fit the random search model
    rf_random.fit(train_features, train_labels)
    print(rf_random.best_params_)
    return rf_random.best_estimator_

#调参：GridSearchCV
from sklearn.model_selection import GridSearchCV
def gridSearchCV(train_features,train_labels):
    # Create the parameter grid based on the results of random search
    param_grid = {
        'bootstrap': [True],
        'max_depth': [20, 25, 30, 35, 40,45,50],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [7, 8, 9,10],
        'n_estimators': [200,250,300,350,400,500]
    }
    # Create a based model
    rf = RandomForestRegressor()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               scoring='neg_root_mean_squared_error', cv=3,
                               n_jobs=-1, verbose=2)
    # Fit the grid search to the data
    grid_search.fit(train_features, train_labels)
    print(grid_search.best_params_)
    return grid_search.best_estimator_


#调参前后的变化
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(test_labels, predictions)))  # 误差越大，该值越大
    print('R_Squared:', r2_score(test_labels, predictions))  # 衡量标准就是正确率，而正确率又在0～1之间，最高百分之百，最低0
    # print('Mean Absolute Error:', mean_absolute_error(test_labels, predictions))  # 误差越大，该值越大


#贝叶斯优化
from skopt import BayesSearchCV
from skopt.space import Real, Integer
def bayesSearchCV(test_features, test_labels):
    # 定义超参数搜索空间
    param_space = {
        'n_estimators': (100, 1000),
        'max_depth': (10, 100),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 10),
        'max_features': ['auto', 'sqrt']
    }



    rf = RandomForestRegressor()
    # 使用BayesSearchCV进行贝叶斯超参数优化
    bayes_search = BayesSearchCV(
        estimator=rf,  # 这是你的估计器
        search_spaces=param_space,  # 使用param_space作为搜索空间
        n_iter=100,  # 迭代次数
        cv=5,       # 交叉验证折数
        n_jobs=-1,  # 使用所有可用的CPU核心
        scoring='neg_mean_squared_error',  # 评分指标，注意使用负的均方误差
        verbose=1,
        random_state=1234
    )

    # 对训练数据进行拟合和优化
    bayes_search.fit(test_features, test_labels.ravel())  # ravel()将y_train从二维数组转换为一维数组

    # 输出最佳超参数配置
    print("Best Parameters:", bayes_search.best_params_)
    return bayes_search.best_params_

##预测值和真实值之间的差异
def predictions_Visualized(actual,predictions,test_labels,outpathfigure):
    #设置图例大小
    plt.rcParams.update({'font.size': 6})
    #plt.legend(loc='upper right')
    # 设置x，y标签大小
    plt.rcParams['font.family'] = ['Times New Roman']
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5), dpi=200)
    label_font = {
        'weight': 'bold',
        'size': 8,
        'family': 'Times New Roman'
    }
    x_axis_data=range(1,len(test_labels)+1)
    y_axis_data1=test_labels
    y_axis_data2=predictions
    # 设置坐标轴选旋转labelrotation=15,及坐标轴大小
    ax.tick_params(axis='x', labelsize=6, bottom=False, labelrotation=15)
    ax.tick_params(axis='y', labelsize=6, bottom=False)
    # 画图
    plt.plot(x_axis_data, y_axis_data1, 'b-', alpha=1, linewidth=1, label='actual')
    plt.plot(x_axis_data, y_axis_data2, 'ro', alpha=1, linewidth=1, label='predictions',markersize='3')
    #plt.plot(x_axis_data, y_axis_data3, 'go--', alpha=0.5, linewidth=1, label='CUBIC')

    plt.legend()  # 显示上面的label
    plt.xlabel('stations_number',label_font)
    plt.ylabel('coherence',label_font)  # accuracy

    # plt.ylim(-1,1)#仅设置y轴坐标范围
    plt.tight_layout(pad=2)
    # 导出图像
    plt.savefig(outpathfigure+filename+'.png',dpi=600,bbox_inches = 'tight')

#特征值和精确度输出excel
def importance_excel(importances,feature_list,RMSE,R2,n,df,filename):
    if(n==0):
        #feature_importances = {feature: round(importance, 2) for feature, importance in zip(feature_list, importances)}
        #feature_importances['RMSE'] = RMSE
        #feature_importances['R2'] = R2
        label = feature_list
        label.append('RMSE')
        label.append('R2')
        label.append('date')
        df=pd.DataFrame(columns=label)
        print(df)
    #new row
    importances.append(RMSE)
    importances.append(R2)
    importances.append(filename)
    # add new row to end of DataFrame
    df.loc[len(df.index)] = importances
    return df

if __name__ == '__main__':
    # 设定文件路径
    path = r'F:\Conherence of InSAR with MachineLearning\Data\自变量和因变量\数据提取\总数据\\'
    # 获取该目录下所有文件，存入列表中
    fileList = os.listdir(path)
    # get_key是sotred函数用来比较的元素，该处用lambda表达式替代函数。
    get_key = lambda i: i.split('.')[0]
    new_sort = sorted(fileList, key=get_key)
    print(fileList, '\n', new_sort)
    n = 0
    # 导出路径
    outpathfigure1 = r'G:\大创数据\Data\outfigure\time\\'
    outpathfigure2 = r'G:\大创数据\Data\outfigure\time_predictions\\'
    outpathexcel = r'F:\Conherence of InSAR with MachineLearning\Data\自变量和因变量\结果\test\\'
    #创建dataframe
    df = pd.DataFrame()
    for i in fileList:
        # 导入数据 并提取标签等
        file = new_sort[n]
        # 导入数据集
        features = pd.read_excel(path+file,engine='openpyxl')
        print(features.head(5))
        # 提取文件名称
        filename = file.split('.', 1)[0]
        print('正在分析的数据为：'+filename)
        # 提取标签
        labels = np.array(features['coherence'])
        #labels[:] = np.nan_to_num(labels)
        # 保留站点名称
        stations_name = np.array((features['zhandian']))
        # Remove the labels from the features
        features = features.drop(['coherence', 'X', 'Y', 'zhandian','year','month','day'], axis=1)
        # Saving feature names for later use
        feature_list = list(features.columns)
        # Convert to numpy array
        #features[:]=np.nan_to_num(features)
        features = np.array(features)

        # 测试
        importances, predictions, train_features, test_features, train_labels, test_labels, RMSE, R2 = randomforenst_regressor(
            features, labels)
        print(importances)
        # predictions_Visualized(labels,predictions,test_labels,outpathfigure2)
        # importance_Visualized(importances, feature_list, outpathfigure1, filename)
        df = importance_excel(importances, feature_list, RMSE, R2, n, df, filename)
        n += 1
        # 保存到本地excel
        writer = pd.ExcelWriter(outpathexcel + '按保护区位置_湿季' + '.xlsx', engine='xlsxwriter')
        # options={'strings_to_urls': False})  # options参数可带可不带，根据实际情况
        df.to_excel(writer, index=False)
        writer._save()
        print('输出完毕！')





    #predictions_Visualized(labels,predictions,test_labels)
    #rf_random=randomizeSearchCV(train_features,train_labels)
    #evaluate(rf_random,test_features, test_labels)
    #rf_grid=gridSearchCV(train_features,train_labels)
    #evaluate(rf_grid, test_features, test_labels)
    #rf_bayes=bayesSearchCV(train_features,train_labels)
    #evaluate(rf_bayes, test_features, test_labels)






