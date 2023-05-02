from yacs.config import CfgNode as CN

_CFG = CN()

_CFG.MODEL = CN()
_CFG.MODEL.META_ARCHITECTURE = 'MulticlassSegmentator'
_CFG.MODEL.DEVICE = "cpu"
_CFG.MODEL.TRAINABLE_LAYERS = 3

# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
_CFG.INPUT = CN()
_CFG.INPUT.IMAGE_SIZE = [512, 512]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_CFG.DATASETS = CN()
_CFG.DATASETS.IMGS_DIR = ''
_CFG.DATASETS.TRAIN_META_PATH = ''
_CFG.DATASETS.VALID_META_PATH = ''

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_CFG.DATA_LOADER = CN()
# Number of data loading threads
_CFG.DATA_LOADER.NUM_WORKERS = 1
_CFG.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_CFG.SOLVER = CN()
# train configs
_CFG.SOLVER.TYPE = 'Adam'
_CFG.SOLVER.MAX_ITER = 2048
_CFG.SOLVER.BATCH_SIZE = 32
_CFG.SOLVER.LR = 1e-3
_CFG.SOLVER.LR_LAMBDA = 0.95

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_CFG.TEST = CN()
_CFG.TEST.BATCH_SIZE = 8

# ---------------------------------------------------------------------------- #
# Output options
# ---------------------------------------------------------------------------- #
_CFG.OUTPUT_DIR = 'outputs/test'