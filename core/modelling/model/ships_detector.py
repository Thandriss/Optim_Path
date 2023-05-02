from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights


class ShipsDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.trainable_layers = max(0, min(5, cfg.MODEL.TRAINABLE_LAYERS))
        self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights, trainable_backbone_layers=self.trainable_layers)

    def export_rebuild(self, target):
        pass

    def forward(self, images):
        return self.model(images)