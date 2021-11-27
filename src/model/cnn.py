import torch
from torch import nn
from timm import create_model

from src.constant import TABULAR_FEATURES


LEN_TABULAR_FEATURES = len(TABULAR_FEATURES)


class PetFinderModel(nn.Module):
    def __init__(self, backbone, pretrained=True, out_dim=1, hidden_size=256, dropout_rate=0.2):
        super(PetFinderModel, self).__init__()
        self.img_layer = create_model(backbone, pretrained=pretrained, num_classes=0)

        self.head = nn.Linear(self.img_layer.num_features + LEN_TABULAR_FEATURES, out_dim)


    def forward(self, img, tabular):
        img_feat = self.img_layer(img)

        x = torch.cat([img_feat, tabular], dim=1)

        x = self.head(x)

        return x, img_feat


if __name__ == '__main__':
    z = torch.randn(4, 3, 380, 380)
    tabular = torch.ones(4, 12)

    # net = PetFinderModel(backbone='tf_efficientnet_b4_ns', hidden_size=8, pretrained=False, out_dim=1)
    #
    # out, _ = net.forward(z, tabular)
    # print(out.size())

    model = create_model('swin_base_patch4_window7_224', num_classes=0)
    print(model.num_features)




