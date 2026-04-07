import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_cifar10_dataloader(batch_size=64, val_ratio=0.1):
    """
    功能：
        加载 CIFAR-10 数据集，并完成训练集、验证集、测试集的预处理与封装

    参数：
        batch_size : 每个批次的样本数
        val_ratio  : 从训练集中划分验证集的比例

    返回：
        train_loader : 训练集数据加载器
        val_loader   : 验证集数据加载器
        test_loader  : 测试集数据加载器
        classes      : CIFAR-10 的类别名称
    """

    # 1. 定义训练集的数据预处理
    # 包括：随机裁剪、随机水平翻转、张量化、标准化
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # 随机裁剪，增强模型泛化能力
        transforms.RandomHorizontalFlip(),      # 随机水平翻转
        transforms.ToTensor(),                  # 转为张量，并归一化到 [0,1]
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),      # CIFAR-10 数据集的均值
            std=(0.2023, 0.1994, 0.2010)        # CIFAR-10 数据集的标准差
        )
    ])

    # 2. 定义测试集/验证集的数据预处理
    # 一般不做随机增强，只做张量化和标准化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    # 3. 下载并加载完整训练集
    full_train_dataset = datasets.CIFAR10(
        root='./data',             # 数据存储路径
        train=True,                # 训练集
        download=True,             # 若本地没有则自动下载
        transform=train_transform  # 使用训练集预处理
    )

    # 4. 下载并加载测试集
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,               # 测试集
        download=True,
        transform=test_transform
    )

    # 5. 从训练集中划分出验证集
    train_size = int((1 - val_ratio) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size]
    )

    # 6. 由于 random_split 后验证集仍继承了训练集的 transform，
    # 这里手动把验证集的 transform 改为测试时的预处理方式
    val_dataset.dataset.transform = test_transform

    # 7. 构建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,      # 训练集通常打乱
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # 8. CIFAR-10 的类别名称
    classes = (
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    )

    return train_loader, val_loader, test_loader, classes


def show_dataset_info(train_loader, val_loader, test_loader, classes):
    """
    输出数据集的基本信息
    """
    print("CIFAR-10 数据集信息如下：")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    print(f"类别数: {len(classes)}")
    print(f"类别名称: {classes}")


if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_cifar10_dataloader(
        batch_size=64,
        val_ratio=0.1
    )

    show_dataset_info(train_loader, val_loader, test_loader, classes)

    # 查看一个 batch 的数据形状
    images, labels = next(iter(train_loader))
    print(f"一个批次图像数据的形状: {images.shape}")   # [batch_size, 3, 32, 32]
    print(f"一个批次标签数据的形状: {labels.shape}")   # [batch_size]