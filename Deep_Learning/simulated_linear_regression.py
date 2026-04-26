import torch
import torch.nn as nn
from torchsummary import summary

class ModelDemo(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=3, out_features=3)
        self.linear2 = nn.Linear(in_features=3, out_features=2)
        self.output = nn.Linear(in_features=2, out_features=2)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.softmax(self.output(x), dim=-1)
        return x

def train():
    my_model = ModelDemo()
    #print(my_model)
    data = torch.randn(size = (5,3))
    #print(data)
    output = my_model(data)
    print(output)
    print(output.shape)
    print(output.requires_grad)
    print('++++++++++++计算模型参数++++++++++++++')
    summary(my_model, input_size=(5,3))
    for name, param in my_model.named_parameters():
        print(name)
        print(param)

if __name__ == '__main__':
    train()
