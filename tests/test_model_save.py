import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('value', torch.Tensor())
        

if __name__ == '__main__':
    m1 = Model()
    m1.value = torch.zeros(10, 10)
    print(m1.state_dict())
    torch.save(m1.state_dict(), 'm.pt')
    m2 = Model()
    m2.load_state_dict(torch.load('m.pt'))
    print(m2.state_dict())
