"""
案例：演示KNN算法，识别图片

介绍：
   每张图片都是由28*28像素组成，即：我们的csv文件中每一行都有784个像素点，表示图片的颜色。
   最终构成图像.
"""

# 导入 matplotlib.pyplot，用于数据可视化（绘图）
import matplotlib.pyplot as plt

# 导入 pandas，用于数据处理和表格数据分析
import pandas as pd
from sklearn.metrics import accuracy_score
# 导入数据集划分函数，用于将数据分为训练集和测试集
from sklearn.model_selection import train_test_split

# 导入 KNN 分类器，用于构建手写数字识别模型
from sklearn.neighbors import KNeighborsClassifier

# 导入 joblib，用于模型的保存与加载（模型持久化）
import joblib

# 从 collections 模块导入 Counter，用于统计类别出现次数（KNN投票机制）
from collections import Counter
# 定义函数：根据索引显示手写数字图片
def show_digit(idx):
    # 读取 CSV 数据（第一列为标签，其余为像素）
    df = pd.read_csv('./dara/手写数字识别.csv')
    # 越界检查（索引必须在合法范围内）
    if idx < 0 or idx > len(df) - 1:
        print('Index out of bounds')
        return
    # 提取特征 x（像素）和标签 y（数字）
    x = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    # 输出当前图片对应的真实标签
    print(f'Label: {y.iloc[idx]}')
    # 输出所有标签的分布情况（计数统计）
    print(f'Label distribution: {Counter(y)}')
    # 输出当前样本的特征形状（应为784维）
    print(f'Shape: {x.iloc[idx].shape}')
    # 将一维像素向量 reshape 为 28×28 图像
    x = x.iloc[idx].values.reshape(28, 28)
    # 显示图像（灰度图）
    plt.imshow(x, cmap='gray')
    plt.axis('off')
    plt.show()

def train_modal():
    df = pd.read_csv('./dara/手写数字识别.csv')
    x = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    print(f'x的形状: {x.shape}')
    print(f'y的形状: {y.shape}')
    # 使用 train_test_split 将数据划分为训练集和测试集（按 8:2 比例）
    # stratify=y 保证训练集和测试集中的类别分布与原数据一致（防止类别不均衡）
    # random_state=42 固定随机种子，确保每次划分结果可复现
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20,stratify=y)
    estimator = KNeighborsClassifier(n_neighbors=5)
    estimator.fit(x_train, y_train)
    print(f'准确率: {estimator.score(x_test, y_test)}')
    print(f'准确率: {accuracy_score(y_test,estimator.predict(x_test))}')
    joblib.dump(estimator, './model/手写数字识别.pkl')
    print('模型保存成功！')

def use_modal():
    img = plt.imread('./dara/demo.png')
   # plt.imshow(img, cmap='gray')
   # plt.axis('off')
   # plt.show()
    estimator = joblib.load('./model/手写数字识别.pkl')
    print(img.shape)
    print(img.reshape(1,-1).shape)
    img = img.reshape(1,-1)
    y_pred = estimator.predict(img)
    print(f'预测解果为:{y_pred}')
# 主程序入口
if __name__ == '__main__':
    train_modal()
    use_modal()


