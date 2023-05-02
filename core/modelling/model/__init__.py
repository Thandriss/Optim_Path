from .multiclass_segmentator import MulticlassSegmentator
from .ships_detector import ShipsDetector

_MODEL_META_ARCHITECTURES = {
    "MulticlassSegmentator": MulticlassSegmentator,
    "ShipsDetector": ShipsDetector,
}

def build_model(cfg):
    meta_arch = _MODEL_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)