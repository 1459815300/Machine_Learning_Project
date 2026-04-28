import torch
import torch.nn as nn

def dm01():
    input_2d = torch.randn(size=(1, 3, 32, 32))
    print(input_2d)

    bn2d = nn.BatchNorm2d(num_features=3,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)

    output = bn2d(input_2d)
    print(output)

def dm02():
    input_1d = torch.randn(size=(32, 32))
    linear = nn.Linear(in_features=32, out_features=32)
    l1 = linear(input_1d)
    print(l1)
    bn1d = nn.BatchNorm1d(num_features=32)
    output1d = bn1d(input_1d)
    print(output1d)

if __name__ == '__main__':
    #dm01()
    dm02()
