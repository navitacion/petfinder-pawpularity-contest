import torch
from torch import nn
from timm import create_model

from src.constant import TABULAR_FEATURES


LEN_TABULAR_FEATURES = len(TABULAR_FEATURES)


def init_weights(m):
    # if type(m) == nn.LazyLinear:
    #     nn.init.xavier_uniform_(m.weight)
    #     nn.init.constant_(m.bias, 0)
    if type(m) == nn.BatchNorm1d:
        nn.init.constant_(m.bias, 0)

# class LinearReluLayer(nn.Module):
#     def __init__(self, hidden_size, out_dims):
#         super(LinearReluLayer, self).__init__()
#
#         self.layer = nn.Sequential(
#             nn.LazyLinear(hidden_size, bias=False),
#             nn.BatchNorm1d(hidden_size),
#             nn.ReLU(inplace=True),
#             nn.LazyLinear(out_dims, bias=False),
#             nn.BatchNorm1d(out_dims)
#         )
#
#         self.layer.apply(init_weights)
#
#     def forward(self, x):
#         x = self.layer(x)
#
#         return x
#
# class TabularModel(nn.Module):
#     def __init__(self, hidden_size):
#         super(TabularModel, self).__init__()
#
#         self.start = LinearReluLayer(hidden_size, hidden_size)
#
#         self.head = nn.ModuleList([
#             LinearReluLayer(hidden_size, hidden_size),
#             LinearReluLayer(hidden_size * 2, hidden_size),
#             LinearReluLayer(hidden_size * 2, hidden_size),
#             LinearReluLayer(hidden_size * 4, hidden_size),
#         ])
#
#         self.last = nn.LazyLinear(hidden_size, bias=True)
#
#     def forward(self, x):
#         x = self.start(x)
#         skip_x = x
#
#         for l in self.head:
#             x = l(x)
#             x = torch.add(x, skip_x)
#             x = nn.ReLU(inplace=True)(x)
#             skip_x = x
#
#         x = self.last(x)
#
#         return x


class PetFinderModel(nn.Module):
    def __init__(self, backbone, pretrained=True, out_dim=1, hidden_size=256, dropout_rate=0.2):
        super(PetFinderModel, self).__init__()
        self.img_layer = create_model(backbone, pretrained=pretrained, num_classes=0)
        # self.tabular_layer = TabularModel(hidden_size)

        self.head = nn.Linear(self.img_layer.num_features + LEN_TABULAR_FEATURES, out_dim)

        nn.init.constant_(self.head.bias, 0.38)


    def forward(self, img, tabular):
        img = self.img_layer(img)
        # tabular = self.tabular_layer(tabular)

        x = torch.cat([img, tabular], dim=1)

        x = self.head(x)

        return x



if __name__ == '__main__':
    z = torch.randn(4, 3, 380, 380)
    tabular = torch.ones(4, 12)

    net = PetFinderModel(backbone='tf_efficientnet_b4_ns', hidden_size=8, pretrained=False, out_dim=1)

    out = net.forward(z, tabular)
    print(out.size())

    # model = create_model('resnet18')
    #
    # print(model)



