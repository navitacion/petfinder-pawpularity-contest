import torch
from torch import nn
from timm import create_model

from src.constant import TABULAR_FEATURES


LEN_TABULAR_FEATURES = len(TABULAR_FEATURES)

class LinearReluLayer(nn.Module):
    def __init__(self, in_features, hidden_size, layer_num=3):
        super(LinearReluLayer, self).__init__()

        layers = []
        for i in range(layer_num):
            if i == 0:
                layers.append(nn.Linear(in_features, hidden_size, bias=False))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size, bias=False))

            layers.append(nn.BatchNorm1d(hidden_size))

            if i == layer_num - 1:
                pass
            else:
                layers.append(nn.ReLU(inplace=True))

        self.layer = nn.ModuleList(layers)

    def forward(self, x):
        for l in self.layer:
            x = l(x)
        return x

class TabularModel(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(TabularModel, self).__init__()

        self.start = nn.Sequential(
            nn.Linear(num_features, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size)
        )

        self.head1 = LinearReluLayer(hidden_size, hidden_size)
        self.head2 = LinearReluLayer(hidden_size, hidden_size)
        self.head3 = LinearReluLayer(hidden_size, hidden_size)
        self.head4 = LinearReluLayer(hidden_size, hidden_size)

    def forward(self, x):
        x1 = self.start(x)

        x2 = self.head1(x1)
        x2 = nn.ReLU()(x2 + x1)

        x3 = self.head2(x2)
        x3 = nn.ReLU()(x3 + x2)

        x4 = self.head3(x3)
        x4 = nn.ReLU()(x4 + x3)

        x5 = self.head4(x4)
        x5 = nn.ReLU()(x5 + x4)

        return x5


class PetFinderModel(nn.Module):
    def __init__(self, backbone, pretrained=True, out_dim=1, hidden_size=256, dropout_rate=0.2):
        super(PetFinderModel, self).__init__()
        self.img_layer = create_model(backbone, pretrained=pretrained, num_classes=0)
        self.tabular_layer = TabularModel(LEN_TABULAR_FEATURES, hidden_size)

        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.img_layer.num_features + hidden_size, out_dim)
        )

    def forward(self, img, tabular):
        img = self.img_layer(img)
        tabular = self.tabular_layer(tabular)

        x = torch.cat([img, tabular], dim=1)

        x = self.head(x)

        return x



if __name__ == '__main__':
    z = torch.randn(4, 3, 380, 380)
    tabular = torch.ones(4, 12)

    net = PetFinderModel(backbone='tf_efficientnet_b4_ns', pretrained=False, out_dim=1)

    out = net(z, tabular)
    print(out.size())

