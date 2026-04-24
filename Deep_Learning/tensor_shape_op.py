import torch

torch.manual_seed(24)
def dm01():
    t1 = torch.randint(1,10,size = (2,3))
    print(f"t1 = {t1}, type(t1) = {type(t1)}")
    print(f"shape(t1) = {t1.shape},row(t1) = {t1.shape[0]},col(t1) = {t1.shape[1]})")
    t2 = t1.reshape(3,2)
    print(f"t2 = {t2}")
    t3 = t1.reshape(6,1)
    print(f"t3 = {t3}")
    t4 = t1.reshape(1,6)
    print(f"t4 = {t4}")
def dm02():
    t1 = torch.randint(1,10,size = (2,3,1))
    print(f"t1 = {t1}, type(t1) = {type(t1)}")
    t2 = t1.unsqueeze(0)
    print(f"t2 = {t2},{t2.shape}")
    t3 = t1.unsqueeze(1)
    print(f"t3 = {t3},{t3.shape}")
    t4 = t1.unsqueeze(-1)
    print(f"t4 = {t4},{t4.shape}")
    t5 = t1.squeeze()
    print(f"t5 = {t5},{t5.shape}")
def dm03():
    x = torch.randint(1,10,size = (2,3,4))
    print(f"x = {x}, type(x) = {type(x)},x(shape) = {x.shape}")
    x1 = x.transpose(1,2)
    print(f"x1 = {x1}, x1.shape = {x1.shape}")
    print(x1.is_contiguous())
    x2 = x1.contiguous()
    print(x2.is_contiguous())
    x3 = x1.permute(1,0,2)
    print(f"x3 = {x3}, x3.shape = {x3.shape}")
    print(x3.is_contiguous())
    if x3.is_contiguous() is True:
        x4 = x3.view(-1, 3)
        print(f"x4 = {x4}, x4.shape = {x4.shape}")
    else:
        print("x3 is not contiguous")
        x4  = x3.contiguous().view(-1, 3)
        print(f"x4 = {x4}, x4.shape = {x4.shape}")
if __name__ == '__main__':
    dm03()