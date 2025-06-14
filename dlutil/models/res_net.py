import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, c_in, subsample=False, c_out=-1):
        super().__init__()
        if not subsample:
            c_out = c_in

        self.block = nn.Sequential(
            nn.Conv2d(
                c_in,
                c_out,
                kernel_size=3,
                padding=1,
                stride=2 if subsample else 1,
                bias=False,
            ),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )

        self.subsample = (
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=2) if subsample else None
        )

    def forward(self, x):
        z = self.block(x)
        if self.subsample is not None:
            x = self.subsample(x)
        out = nn.ReLU(x + z)
        return out


class ResNet(nn.Module):
    def __init__(self, num_class=10, block_nums=[3, 3, 3], c_hidden=[16, 32, 64]):
        super().__init__()
        self.input_net = nn.Conv2d(3, c_hidden[0], kernel_size=3, padding=1)

        layers = []
        for stage_id, c in enumerate(c_hidden):
            block_num = block_nums[stage_id]
            for block_id in range(block_num):
                subsample = block_id == 0 and stage_id > 0
                layers.append(
                    ResBlock(c, subsample=subsample, c_out=c * 2 if subsample else c)
                )

        self.stages = nn.Sequential(*layers)

        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_hidden[-1], num_class),
        )

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = self.input_net(x)
        x = self.stages(input)
        x = self.output_net(x)
        return x
