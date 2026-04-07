"""
案例：演示KNN算法，识别图片

介绍：
   每张图片都是由28*28像素组成，即：我们的csv文件中每一行都有784个像素点，表示图片的颜色。
   最终构成图像.
"""
#导入绘图库，用于显示图像
import matplotlib.pyplot as plt

#导入pandas，用于读取和处理数据
import pandas as pd

#导入准确率评估函数
from sklearn.metrics import accuracy_score

#导入数据集划分函数
from sklearn.model_selection import train_test_split

#导入KNN分类器
from sklearn.neighbors import KNeighborsClassifier

#导入模型保存工具
import joblib

#导入Counter用于统计类别分布
from collections import Counter

#定义函数：根据索引显示手写数字图像
def show_digit(idx):
    #读取CSV数据（第1列为标签，其余为像素）
    df=pd.read_csv('./dara/手写数字识别.csv')
    #判断索引是否越界
    if idx<0 or idx>len(df)-1:
        print('Index out of bounds')
        return
    #提取特征和标签
    x=df.iloc[:,1:]
    y=df.iloc[:,0]
    #输出标签信息
    print(f'Label:{y.iloc[idx]}')
    #输出标签分布
    print(f'Label distribution:{Counter(y)}')
    #输出当前样本维度
    print(f'Shape:{x.iloc[idx].shape}')
    #重塑为28×28图像
    x=x.iloc[idx].values.reshape(28,28)
    #显示图像
    plt.imshow(x,cmap='gray')
    plt.axis('off')
    plt.show()

#定义函数：训练模型
def train_modal():
    #读取数据
    df=pd.read_csv('./dara/手写数字识别.csv')
    #划分特征和标签
    x=df.iloc[:,1:]
    y=df.iloc[:,0]
    #输出数据形状
    print(f'x的形状:{x.shape}')
    print(f'y的形状:{y.shape}')
    #划分训练集和测试集（8:2）
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=20,stratify=y)
    #创建KNN模型（K=5）
    estimator=KNeighborsClassifier(n_neighbors=5)
    #训练模型
    estimator.fit(x_train,y_train)
    #输出模型准确率（两种方式）
    print(f'准确率:{estimator.score(x_test,y_test)}')
    print(f'准确率:{accuracy_score(y_test,estimator.predict(x_test))}')
    #保存模型
    joblib.dump(estimator,'./model/手写数字识别.pkl')
    print('模型保存成功！')

#定义函数：加载模型并预测
def use_modal():
    #读取测试图片
    img=plt.imread('./dara/demo.png')
    # 如果是RGB图，转灰度
    if len(img.shape) == 3:
        img = img[:, :, 0]
    #加载已训练模型
    estimator=joblib.load('./model/手写数字识别.pkl')
    #输出图片原始形状
    print(img.shape)
    #调整为模型输入格式（1×784）
    print(img.reshape(1,-1).shape)
    img=img.reshape(1,-1)
    #进行预测
    y_pred=estimator.predict(img)
    print(f'预测结果为:{y_pred}')

#程序入口
if __name__=='__main__':
    #训练模型
    train_modal()
    #使用模型预测
    use_modal()

