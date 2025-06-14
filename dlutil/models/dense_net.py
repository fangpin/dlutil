import torch.nn as nn
import torch


class DenseLayer(nn.Module):
    def __init__(self, c_in, bn_factor, growth_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            nn.ReLU(),
            nn.Conv2d(c_in, bn_factor * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(bn_factor * growth_rate),
            nn.ReLU(),
            nn.Conv2d(
                bn_factor * growth_rate,
                growth_rate,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, x):
        z = self.net(x)
        out = torch.cat([x, z], dim=1)
        return out


class DenseBlock(nn.Module):
    def __init__(self, num_layers, c_in, bn_factor, growth_rate):
        super().__init__()
        layers = []
        for idx in range(num_layers):
            layers.append(DenseLayer(c_in + idx * growth_rate, bn_factor, growth_rate))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        out = self.blocks(x)
        return out


class TransitionLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            nn.ReLU(),
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class DenseNet(nn.Module):
    def __init__(
        self, num_class=10, growth_rate=16, bn_factor=2, num_layers=[6, 6, 6, 6]
    ):
        super().__init__()
        c_hidden = growth_rate * bn_factor
        self.input_net = nn.Sequential(nn.Conv2d(3, c_hidden, kernel_size=3, padding=1))

        stages = []
        for stage_id, num_layer in enumerate(num_layers):
            stages.append(
                DenseBlock(
                    num_layers=num_layer,
                    c_in=c_hidden,
                    bn_factor=bn_factor,
                    growth_rate=growth_rate,
                )
            )
            c_hidden = c_hidden + num_layer * growth_rate
            if stage_id + 1 < len(num_layers):
                stages.append(TransitionLayer(c_in=c_hidden, c_out=c_hidden // 2))
                c_hidden = c_hidden // 2
        self.stages_net = nn.Sequential(*stages)

        self.output_net = nn.Sequential(
            nn.BatchNorm2d(c_hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_hidden, num_class),
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
        x = self.input_net(x)
        x = self.stages_net(x)
        x = self.output_net(x)
        return x
