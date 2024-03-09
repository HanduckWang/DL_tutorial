from torch import nn


class Duck(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = input + 1
        return output

duck = Duck()
