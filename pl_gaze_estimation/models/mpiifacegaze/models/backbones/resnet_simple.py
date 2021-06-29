import torch
import torchvision
import torchvision.models.utils
from omegaconf import DictConfig


class Network(torchvision.models.ResNet):
    def __init__(self, config: DictConfig):
        block_name = config.MODEL.BACKBONE.RESNET_BLOCK
        if block_name == 'basic':
            block = torchvision.models.resnet.BasicBlock
        elif block_name == 'bottleneck':
            block = torchvision.models.resnet.Bottleneck
        else:
            raise ValueError
        layers = list(config.MODEL.BACKBONE.RESNET_LAYERS) + [1]
        super().__init__(block, layers)
        del self.layer4
        del self.avgpool
        del self.fc

        pretrained_name = config.MODEL.BACKBONE.PRETRAINED
        if pretrained_name:
            state_dict = torchvision.models.utils.load_state_dict_from_url(
                torchvision.models.resnet.model_urls[pretrained_name])
            self.load_state_dict(state_dict, strict=False)

        with torch.no_grad():
            n_channels = config.DATASET.N_CHANNELS
            image_size = config.DATASET.IMAGE_SIZE
            data = torch.zeros((1, n_channels, image_size, image_size),
                               dtype=torch.float32)
            features = self.forward(data)
            self.n_features = features.shape[1]
            self.feature_map_size = features.shape[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
