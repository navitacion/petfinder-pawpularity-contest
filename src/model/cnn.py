import torch
from torch import nn
from timm import create_model

from src.constant import TABULAR_FEATURES


LEN_TABULAR_FEATURES = len(TABULAR_FEATURES)


class LinearReluLayer(nn.Module):
    def __init__(self, hidden_size, layer_num=3, dropout_rate=0.5):
        super(LinearReluLayer, self).__init__()

        layers = []
        for i in range(layer_num):
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_size, hidden_size, bias=False))

        layers = layers[:-1]
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_size, hidden_size, bias=False))

        self.layer = nn.ModuleList(layers)

    def forward(self, x):
        for l in self.layer:
            x = l(x)
        return x


class TabularModel(nn.Module):
    def __init__(self, num_features, hidden_size, dropout_rate, num_unit=4):
        super(TabularModel, self).__init__()

        self.start = nn.Linear(num_features, hidden_size, bias=False)

        self.head = nn.ModuleList([
            LinearReluLayer(hidden_size, dropout_rate=dropout_rate) for _ in range(num_unit)
        ])

    def forward(self, x):
        x = self.start(x)
        skip_x = x

        for l in self.head:
            x = l(x) + skip_x
            skip_x = x

        return x


class PetFinderModel(nn.Module):
    def __init__(self, backbone, pretrained=True, out_dim=1, hidden_size=256, dropout_rate=0.2):
        super(PetFinderModel, self).__init__()
        self.img_layer = create_model(backbone, pretrained=pretrained, num_classes=0)
        self.tabular_layer = TabularModel(LEN_TABULAR_FEATURES, hidden_size, dropout_rate)

        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.img_layer.num_features + hidden_size, out_dim)
        )

    def forward(self, img, tabular):
        img = self.img_layer(img)
        tabular = self.tabular_layer(tabular)

        x = torch.cat([img, tabular], dim=1)

        x = self.head(x)

        return x, img



if __name__ == '__main__':
    z = torch.randn(4, 3, 380, 380)
    tabular = torch.ones(4, 12)

    net = PetFinderModel(backbone='tf_efficientnet_b4_ns', pretrained=False, out_dim=1)

    out, _ = net.forward(z, tabular)
    print(out.size())

    # model = create_model("swin_large_patch4_window12_384", pretrained=False)
    # # model.head = nn.Linear(model.head.in_features, 128)
    # print(model.head)




