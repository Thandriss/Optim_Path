import os
import torch
import logging
import argparse
import cv2 as cv
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from core.config import cfg
from core.utils.logger import setup_logger
from core.utils import dist_util
from core.modelling.model import build_model
from core.utils.checkpoint import CheckPointer
from core.data.transforms import transforms2


def load_model(cfg, args):
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


def main() -> int:
    # Create argument parser
    parser = argparse.ArgumentParser(description='Dirt Blockage Export With PyTorch')
    parser.add_argument('--model-path', dest='model_path', required=True, type=str, default="", metavar="FILE",
                        help="path to onnx model")
    parser.add_argument('--input-img', dest='input_img', required=True, type=str, default="", metavar="FILE",
                        help="path to input image")
    parser.add_argument('--output-img', dest='output_img', required=True, type=str, default="", metavar="FILE",
                        help="path to output image")
    parser.add_argument('--scale', dest='scale', required=False, type=float, default=1.0,
                        help="Image scaling factor")
    parser.add_argument('--tile-size', dest='tile_size', required=False, type=int, default=512,
                        help="Tile size")
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()

    # Create logger
    logger = setup_logger("INFER", dist_util.get_rank())
    logger.info(args)

    # Create model
    print('Onnx session running on {0} device...'.format(ort.get_device()))
    # ort_session = ort.InferenceSession(args.model_path, providers=['CUDAExecutionProvider'])
    ort_session = ort.InferenceSession(args.model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # Read input image
    image = cv.imread(args.input_img)
    if image is None:
        logger.error("Failed to read input image file")

    # Resize images
    if not np.isclose(args.scale, 1.0):
        image = cv.resize(image, dsize=None, fx=args.scale, fy=args.scale, interpolation=cv.INTER_AREA)       

    # Prepare expanded sources
    result_img_height = ((image.shape[0] // args.tile_size) + 1) * args.tile_size
    result_img_width = ((image.shape[1] // args.tile_size) + 1) * args.tile_size
    image_expanded = np.zeros((result_img_height, result_img_width, 3), np.uint8)
    image_expanded[:image.shape[0], :image.shape[1]] = image

    # Prepare result mask
    result_mask = np.zeros((result_img_height, result_img_width, 1), np.uint8)

    # Infer model
    pbar = tqdm(total=((image.shape[0] // args.tile_size) + 1)*((image.shape[1] // args.tile_size) + 1))

    for y_offset in range(0, image_expanded.shape[0], args.tile_size):
        for x_offset in range(0, image_expanded.shape[1], args.tile_size):

            image_tile = image_expanded[y_offset:y_offset + args.tile_size, x_offset:x_offset + args.tile_size]

            with torch.no_grad():
                input, _, _ = transforms2.ConvertFromInts()(image_tile)
                input, _, _ = transforms2.Normalize()(input)
                # input, _, _ = transforms2.Standardize()(input)
                input, _, _ = transforms2.ToTensor(norm_label=False)(input)

                input = input.unsqueeze(0)
                # outputs = model(input.to(device))
                # outputs = torch.softmax(outputs, dim=1).argmax(dim=1)
                images_np = input.numpy().astype(np.float32)
                ort_inputs = {ort_session.get_inputs()[0].name: images_np}
                ort_outs = ort_session.run(None, ort_inputs)
                ort_outs = np.array(ort_outs)
                ort_outs = np.array(ort_outs)
                # ort_outs = np.array(ort_outs, dtype=np.float32)
                ort_outs = ort_outs.transpose((0,1,2,3,4))
                outputs = ort_outs.squeeze(0) #ort_outs.squeeze(0)
                outputs = outputs.squeeze(0)
                outputs = outputs[0]
                outputs = np.reshape(outputs, [512,512,1])
                result_mask[y_offset:y_offset + args.tile_size, x_offset:x_offset + args.tile_size] = outputs

                pbar.update(1)

    pbar.close()

    # Save colored result
    mask = result_mask[0:image.shape[0], 0:image.shape[1]]

    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[np.squeeze(mask == 1, -1), :] = (255, 128, 0) # river
    # color_mask[np.squeeze(mask == 2, -1), :] = (0, 255, 239) # water
    result = cv.addWeighted(image, 0.9, color_mask, 0.1, 1.0)

    filepath = os.path.abspath(args.output_img)
    print("Saving result to '{0}'...".format(filepath))
    cv.imwrite(filepath, result)


    print("Done.")
    return 0


if __name__ == '__main__':
    exit(main())
