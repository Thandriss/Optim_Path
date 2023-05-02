import os
import torch
import logging
import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm
from core.config import cfg
from core.utils.logger import setup_logger
from core.utils import dist_util
from core.modelling.model import build_model
from core.utils.checkpoint import CheckPointer
from core.data.transforms import transforms2


def load_model(cfg):
    logger = logging.getLogger('INFER')

    # Create device
    device = torch.device(cfg.MODEL.DEVICE)

    # Create model
    model = build_model(cfg)
    model.to(device)

    optimizer = None
    scheduler = None

    # Create checkpointer
    arguments = {"epoch": 0}
    save_to_disk = dist_util.get_rank() == 0
    checkpointer = CheckPointer(model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk, logger)
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    return model

def class_land(config_file, input_img, output_img, scale = 1.0, opts = []):
    # Create config
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()

    # Create logger
    logger = setup_logger("INFER", dist_util.get_rank(), cfg.OUTPUT_DIR)
    # logger.info(args)
    logger.info("Loaded configuration file {}".format(config_file))

    # Create model
    model = load_model(cfg)
    model.eval()

    # Create device
    device = torch.device(cfg.MODEL.DEVICE)

    # Read input image
    image = cv.imread(input_img)
    if image is None:
        logger.error("Failed to read input image file")

    # Resize images
    if not np.isclose(scale, 1.0):
        image = cv.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)

        # Prepare expanded sources
    tile_size = cfg.INPUT.IMAGE_SIZE[0]  # TODO: w, h
    result_img_height = ((image.shape[0] // tile_size) + 1) * tile_size
    result_img_width = ((image.shape[1] // tile_size) + 1) * tile_size
    image_expanded = np.zeros((result_img_height, result_img_width, 3), np.uint8)
    image_expanded[:image.shape[0], :image.shape[1]] = image

    # Prepare result mask
    result_mask = np.zeros((result_img_height, result_img_width, 1), np.uint8)

    # Infer model
    pbar = tqdm(total=((image.shape[0] // tile_size) + 1) * ((image.shape[1] // tile_size) + 1))

    for y_offset in range(0, image_expanded.shape[0], tile_size):
        for x_offset in range(0, image_expanded.shape[1], tile_size):
            image_tile = image_expanded[y_offset:y_offset + tile_size, x_offset:x_offset + tile_size]

            with torch.no_grad():
                input, _, _ = transforms2.ConvertFromInts()(image_tile)
                input, _, _ = transforms2.Normalize()(input)
                # input, _, _ = transforms2.Standardize()(input)
                input, _, _ = transforms2.ToTensor(norm_label=False)(input)

                input = input.unsqueeze(0)
                outputs = model(input.to(device))
                outputs = torch.softmax(outputs, dim=1).argmax(dim=1)
                result_mask[y_offset:y_offset + tile_size, x_offset:x_offset + tile_size, 0] = outputs.to('cpu')
                pbar.update(1)

    pbar.close()

    # Save colored result
    mask = result_mask[0:image.shape[0], 0:image.shape[1]]

    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[np.squeeze(mask == 1, -1), :] = (255, 255, 0)  # (255, 128, 0) # urban_land
    color_mask[np.squeeze(mask == 2, -1), :] = (0, 255, 255)  # (0, 255, 239) # agriculture_land
    color_mask[np.squeeze(mask == 3, -1), :] = (255, 0, 255)  # (0, 255, 239) # rangeland
    color_mask[np.squeeze(mask == 4, -1), :] = (0, 255, 0)  # (0, 255, 239) # forest_land
    color_mask[np.squeeze(mask == 5, -1), :] = (255, 0, 0)  # (0, 255, 239) # water
    color_mask[np.squeeze(mask == 6, -1), :] = (255, 255, 255)  # (0, 255, 239) # barren_land
    color_mask[np.squeeze(mask == 7, -1), :] = (0, 0, 0)  # (0, 255, 239) # unknown
    result = cv.addWeighted(image, 0.0, color_mask, 1.0, 1.0)

    filepath = os.path.abspath(output_img)
    print("Saving result to '{0}'...".format(filepath))
    cv.imwrite(filepath, result)
    print("Done.")
    return result

def main() -> int:
    # Create argument parser
    parser = argparse.ArgumentParser(description='Dirt Blockage Export With PyTorch')
    parser.add_argument('--cfg', dest='config_file', required=True, type=str, default="", metavar="FILE",
                        help="path to config file")
    parser.add_argument('--input-img', dest='input_img', required=True, type=str, default="", metavar="FILE",
                        help="path to input image")
    parser.add_argument('--output-img', dest='output_img', required=True, type=str, default="", metavar="FILE",
                        help="path to output image")
    parser.add_argument('--scale', dest='scale', required=False, type=float, default=1.0,
                        help="Image scaling factor")
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()
    print(args.opts)
    print(args.scale)
    class_land(args.config_file, args.input_img, args.output_img, args.scale, args.opts)
    return 0


if __name__ == '__main__':
    exit(main())
