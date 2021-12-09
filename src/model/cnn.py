import torch
from torch import nn
from timm import create_model

from src.constant import TABULAR_FEATURES


LEN_TABULAR_FEATURES = len(TABULAR_FEATURES)


class PetFinderModel(nn.Module):
    def __init__(self, backbone, pretrained=True, out_dim=1, dropout_rate=0.0):
        super(PetFinderModel, self).__init__()
        self.img_layer = create_model(backbone, pretrained=pretrained, num_classes=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(self.img_layer.num_features + LEN_TABULAR_FEATURES, out_dim)


    def forward(self, img, tabular):
        img_feat = self.img_layer(img)
        x = self.dropout(img_feat)
        x = torch.cat([x, tabular], dim=1)

        x = self.head(x)

        return x, img_feat


class PetFinderImageModel(nn.Module):
    def __init__(self, backbone, pretrained=True, out_dim=1, dropout_rate=0.0):
        super(PetFinderImageModel, self).__init__()
        self.backbone = create_model(backbone, pretrained=pretrained, num_classes=out_dim)

    def forward(self, img, tabular=None):
        return self.backbone(img)


if __name__ == '__main__':
    z = torch.randn(4, 3, 380, 380)
    tabular = torch.ones(4, 12)

    net = PetFinderModel(backbone='tf_efficientnet_b4_ns', pretrained=False, out_dim=1)

    out, _ = net.forward(z, tabular)
    print(out.size())

    # model = create_model("swin_large_patch4_window12_384", pretrained=False)
    # # model.head = nn.Linear(model.head.in_features, 128)
    # print(model.head)




