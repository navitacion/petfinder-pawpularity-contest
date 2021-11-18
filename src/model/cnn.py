import torch
from torch import nn
from timm import create_model

from src.constant import TABULAR_FEATURES


LEN_TABULAR_FEATURES = len(TABULAR_FEATURES)

class LinearReluLayer(nn.Module):
    def __init__(self, in_features, out_dims):
        super(LinearReluLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_dims),
            nn.ReLU(inplace=True),
            nn.Linear(out_dims, out_dims),
            nn.ReLU(inplace=True),
            nn.Linear(out_dims, out_dims),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)

class TabularModel(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(TabularModel, self).__init__()

        self.head1 = LinearReluLayer(num_features, hidden_size)
        self.head2 = LinearReluLayer(hidden_size + num_features, hidden_size)
        self.head3 = LinearReluLayer(hidden_size * 2 + num_features, hidden_size)
        self.head4 = LinearReluLayer(hidden_size * 3 + num_features, hidden_size)

    def forward(self, x):
        x1 = self.head1(x)
        x = torch.cat([x, x1], dim=1)
        x2 = self.head2(x)
        x = torch.cat([x, x2], dim=1)
        x3 = self.head3(x)
        x = torch.cat([x, x3], dim=1)
        x4 = self.head4(x)

        return x4


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

