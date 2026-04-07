"""
example : Show grid search and cross-validation
Cross-validation
原理：
把数据分成n份，例如分为4份
   第一次，把第一份数据作为验证集（测试集），其他作为训练集，训练模型，预测模型，获取：准确率 ——> 准确率1
   第二次，把第二份数据作为验证集（测试集），其他作为训练集，训练模型，预测模型，获取：准确率 ——> 准确率2
   第三次，把第三份数据作为验证集（测试集），其他作为训练集，训练模型，预测模型，获取：准确率 ——> 准确率3
   第四次，把第四份数据作为验证集（测试集），其他作为训练集，训练模型，预测模型，获取：准确率 ——> 准确率4
   然后计算上述4次准确率的平均值，作为：模型最终的准确率

   假设第四次最好（准确率最高），则：用全部数据（训练集 + 测试集）训练模型，再次用测试集对模型测试。
目的：
   为了让模型的最终验证结果更准确

Grid Search
目的/作用：
   寻找最优超参数
原理：
    接受超参数可能出现的值，然后针对于超参每个值进行交叉验证，最终得到最佳超参组合
超参数：
    需要用户手动录入的数据，不同的超参（组合），可能会影响最终评测结果。
    调用GridSearchCV这个API
"""# 导入鸢尾花数据集（经典分类数据集）
from sklearn.datasets import load_iris

# 导入数据集划分工具、网格搜索
from sklearn.model_selection import train_test_split, GridSearchCV

# 数据标准化（归一化的一种）
from sklearn.preprocessing import StandardScaler

# KNN 分类器
from sklearn.neighbors import KNeighborsClassifier

# 模型评估指标（准确率）
from sklearn.metrics import accuracy_score


# ===================== 1. 加载数据 =====================
iris_data = load_iris()
# iris_data.data   -> 特征 (X)
# iris_data.target -> 标签 (y)


# ===================== 2. 划分训练集和测试集 =====================
x_train, x_test, y_train, y_test = train_test_split(
    iris_data.data,         # 特征
    iris_data.target,       # 标签
    test_size=0.2,          # 测试集占20%
    random_state=22         # 固定随机种子，保证结果可复现
)


# ===================== 3. 数据标准化 =====================
transfer = StandardScaler()

# 用训练集计算均值和方差，并进行标准化
x_train = transfer.fit_transform(x_train)

# 用同样的参数去转换测试集（⚠️不能fit）
x_test = transfer.transform(x_test)


# ===================== 4. 构建模型 =====================
estimator = KNeighborsClassifier()


# ===================== 5. 设置超参数搜索空间 =====================
param_dict = {
    'n_neighbors': [i for i in range(1, 11)]  # K值从1到10
}


# ===================== 6. 网格搜索 + 交叉验证 =====================
estimator = GridSearchCV(
    estimator,     # 原始模型
    param_dict,    # 参数组合
    cv=4           # 4折交叉验证
)

# 在训练集上进行训练 + 调参
estimator.fit(x_train, y_train)


# ===================== 7. 输出最优结果 =====================
print(f'最优评分:{estimator.best_score_}')
# 👉 交叉验证中的平均最优准确率

print(f'最优超参数组合:{estimator.best_params_}')
# 👉 最优的 K 值

print(f'最优的估计器对象:{estimator.best_estimator_}')
# 👉 已经训练好的最佳模型

print(f'具体的交叉验证结果:{estimator.cv_results_}')
# 👉 所有参数组合的详细结果（字典，包含均值、方差等）


# ===================== 8. （⚠️这里你写法有问题） =====================
# 你这里手动指定了 n_neighbors=3，相当于“丢弃了刚才调参的结果”

# 正确写法应该是：
# estimator = estimator.best_estimator_

# 但你现在写的是：
estimator = KNeighborsClassifier(n_neighbors=3)


# ===================== 9. 重新训练模型 =====================
estimator.fit(x_train, y_train)


# ===================== 10. 预测 =====================
y_pred = estimator.predict(x_test)


# ===================== 11. 评估模型 =====================
print(f'准确率:{accuracy_score(y_test, y_pred)}')