import os
import json
import codecs
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset


class ShipDataset(Dataset):
    def __init__(self, imgs_dir:str, meta_path:str, transforms=None):
        self.imgs_dir = imgs_dir
        json_data = codecs.open(meta_path, 'r').read()
        self.meta = json.loads(json_data)
        self.max_rects = len(max(self.meta["data"], key=lambda item: len(item["rects"]))['rects'])
        self.transforms = transforms
        self.dummy_mask = None

    def __len__(self):
        return len(self.meta["data"])

    def __getitem__(self, idx):
        meta_item = self.meta["data"][idx]

        # Read image
        imgpath = os.path.join(self.imgs_dir, meta_item["filename"])
        image = cv.imread(imgpath)

        # Create mask
        if self.dummy_mask is None:
            self.dummy_mask = np.full_like(image, 255, dtype=np.uint8)
        mask = self.dummy_mask

        # Read bounding rects (from [x, y, w, h] to [x1, y1, x2, y2])
        rects = np.array([[r[0], r[1], r[0]+r[2], r[1]+r[3]] for r in meta_item["rects"]], dtype=np.int)

        # Pad rects array to max size
        rects_real_num = rects.shape[0]
        assert rects_real_num in range(0, self.max_rects + 1)
        if rects_real_num == self.max_rects:
            pass
        elif rects_real_num == 0:
            rects = np.zeros(shape=(self.max_rects, 4), dtype=np.int)
        else:
            dummy_rects = np.zeros(shape=(self.max_rects - rects_real_num, 4), dtype=np.int)
            rects = np.concatenate([rects, dummy_rects])
        assert rects.shape == (self.max_rects, 4)

        # Prepare data
        if self.transforms:
            image, rects, mask = self.transforms(image, rects, mask)

        return image, rects, mask, rects_real_num

    def visualize(self, tick_ms=25):
        for i in range(0, self.__len__()):
            image, rects, mask, rects_real_num = self.__getitem__(i)

            rects = rects[:rects_real_num]

            for r in rects:
                cv.rectangle(image, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 1)

            cv.imshow('Image', image.astype(np.uint8))
            cv.imshow('Mask', mask.astype(np.uint8))
            if cv.waitKey(tick_ms) & 0xFF == ord('q'):
                return
