"""
案例：通过KNN算法实现 鸢尾花 的分类操作
"""
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


#1.定义函数，加载数据集，查看数据集
def dm01_load_iris():
    iris_data= load_iris()
    #print(f'数据集:{iris_data}')
    print(f'数据集的类型:{type(iris_data)}')
    print(f'数据集的数据条目:{len(iris_data.data)}')
    print(f'数据集所有的键:{iris_data.keys()}')
    print(f'数据集的frame:{iris_data.frame}')
    print(f'数据集所有的标签名字:{iris_data.target_names}')
    #print(f'数据集所有的描述信息:{iris_data.DESCR}')
    print(f'数据集所有的特征名字:{iris_data.feature_names}')
def dm02_show_iris():
    iris_data= load_iris()
    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    iris_df['label'] = iris_data.target
    sns.lmplot(data=iris_df, x='sepal length (cm)', y='sepal width (cm)', hue='label',fit_reg=False)
    plt.title('Iris_Database')
    plt.tight_layout()
    plt.show()
def dm03_split_test_train():
    iris_data= load_iris()
    train_test_split(iris_data.data, iris_data.target, test_size=0.2,random_state=21)

if __name__ == '__main__':
    #dm01_load_iris()
    #dm02_show_iris()
    #dm03_split_test_train()