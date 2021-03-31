
import torch

from torch import nn

# linear = nn.Linear(c2, c2)

if __name__ =="__main__":

    x = torch.randn([1,3,16,16])

    p = x.flatten(2)  # squeeze
    p = p.unsqueeze(0)
    p = p.transpose(0, 3)
    p = p.squeeze(3)
    print(p)
    # x = p + e