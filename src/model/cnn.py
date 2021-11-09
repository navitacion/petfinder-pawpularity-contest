import torch
from torch import nn
from timm import create_model

from src.constant import TABULAR_FEATURES


LEN_TABULAR_FEATURES = len(TABULAR_FEATURES)

class Timm_model(nn.Module):
    def __init__(self, backbone, pretrained=True, out_dim=5):
        super(Timm_model, self).__init__()
        self.base = create_model(backbone, pretrained=pretrained, num_classes=out_dim)

    def forward(self, x):
        return self.base(x)



class TabularModel(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(TabularModel, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.head(x)


class PetFinderModel(nn.Module):
    def __init__(self, backbone, pretrained=True, out_dim=1, hidden_size=256):
        super(PetFinderModel, self).__init__()
        self.img_layer = create_model(backbone, pretrained=pretrained, num_classes=0)
        self.tabular_layer = TabularModel(LEN_TABULAR_FEATURES, hidden_size)

        self.head = nn.Sequential(
            nn.Linear(self.img_layer.num_features + hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, out_dim)
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

