import os
import sys
import shutil
import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm
from glob import glob


def convert_dataset(src_root:str, dst_root:str, tile_size:int, scale:float=1.0, skip_empty:bool=False):
    sample_paths = sorted(glob(os.path.join(src_root, '*')))

    for sample_path in tqdm(sample_paths):
        sample_name = os.path.basename(os.path.normpath(sample_path))

        # Read image
        img_paths = glob(os.path.join(sample_path, 'img/*'))
        if len(img_paths) != 1:
            print("Found more than 1 images in '{0}'. Skipped.".format(sample_path))
            continue
   
        image_orig = cv.imread(img_paths[0])
        if image_orig is None:
            print("Failed to read image '{0}'. Skipped.".format(img_paths[0]))
            continue

        # Read 'road' mask
        road_paths = glob(os.path.join(sample_path, 'mask_road/*'))
        if len(road_paths) != 1:
            print("Found more than 1 road masks in '{0}'. Skipped.".format(sample_path))
            continue

        road_orig = cv.imread(road_paths[0])
        if road_orig is None:
            print("Failed to read road mask '{0}'. Skipped.".format(road_paths[0]))
            continue
        road_orig = cv.cvtColor(road_orig, cv.COLOR_BGR2GRAY)


        # Check shapes
        if not (image_orig.shape[0:2] == road_orig.shape[0:2]):
            print("Shapes of images are inconsistent in '{0}'. Skipped.".format(sample_path))
            continue

        # Resize images
        if not np.isclose(scale, 1.0):
            image_orig = cv.resize(image_orig, dsize=None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
            road_orig = cv.resize(road_orig, dsize=None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
            # water_orig = cv.resize(water_orig, dsize=None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)

        # Create ROI
        roi = np.full(shape=(image_orig.shape[0], image_orig.shape[1], 1), dtype=np.uint8, fill_value=255)

        # Prepare expanded sources
        height_exp = ((image_orig.shape[0] // tile_size) + 1) * tile_size
        width_exp = ((image_orig.shape[1] // tile_size) + 1) * tile_size
        image_exp = np.zeros((height_exp, width_exp, 3), np.uint8)
        image_exp[:image_orig.shape[0], :image_orig.shape[1]] = image_orig
        road_exp = np.zeros((height_exp, width_exp), np.uint8)
        road_exp[:road_orig.shape[0], :road_orig.shape[1]] = road_orig
        roi_exp = np.zeros((height_exp, width_exp, 1), dtype=roi.dtype)
        roi_exp[:roi.shape[0], :roi.shape[1]] = roi

        # Prepare output dirs
        output_dir = os.path.join(dst_root, sample_name)
        
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)

        output_dir_img = os.path.join(output_dir, 'images')
        output_dir_label = os.path.join(output_dir, 'labels')
        output_dir_rois = os.path.join(output_dir, 'rois')

        os.makedirs(output_dir_img, exist_ok=True)
        os.makedirs(output_dir_label, exist_ok=True)
        os.makedirs(output_dir_rois, exist_ok=True)

        # Split sources to tiles
        saved_counter = 0
        for tile_y_idx, y_offset in enumerate(range(0, height_exp, tile_size)):
            for tile_x_idx, x_offset in enumerate(range(0, width_exp, tile_size)):

                image_tile = image_exp[y_offset:y_offset + tile_size, x_offset:x_offset + tile_size]

                # Check if image empty
                black_count = np.count_nonzero(image_tile <= 0) / (image_tile.size)
                white_count = np.count_nonzero(image_tile >= 255) / (image_tile.size)
                if skip_empty and (black_count > 0.5 or white_count > 0.5):
                    continue

                # Create label
                label_tile = np.zeros((tile_size, tile_size), np.uint8)
                road_tile = road_exp[y_offset:y_offset + tile_size, x_offset:x_offset + tile_size]
                label_tile[road_tile > 0] = 1

                # Skip empty labels
                if skip_empty and np.count_nonzero(label_tile) < 1:
                    continue

                # Save image
                image_filename = os.path.join(output_dir_img, f'img_{tile_y_idx}_{tile_x_idx}.png')
                cv.imwrite(image_filename, image_tile)

                # Save label
                label_filename = os.path.join(output_dir_label, f'label_{tile_y_idx}_{tile_x_idx}.png')
                cv.imwrite(label_filename, label_tile)

                # Save roi
                roi_filename = os.path.join(output_dir_rois, f'roi_{tile_y_idx}_{tile_x_idx}.png')
                roi_tile = roi_exp[y_offset:y_offset + tile_size, x_offset:x_offset + tile_size]
                cv.imwrite(roi_filename, roi_tile)


                saved_counter += 1

        if saved_counter < 1:
            shutil.rmtree(output_dir)


def main() -> int:
    # Create argument parser
    parser = argparse.ArgumentParser(description='Dataset creator for training')
    parser.add_argument('--src-root', dest='src_root', required=True, type=str, default="",
                        help="Path to dataset root directory")
    parser.add_argument('--dst-root', dest='dst_root', required=True, type=str, default="",
                        help="Path where to save converted dataset")
    parser.add_argument('--tile-size', dest='tile_size', required=False, type=int, default=512,
                        help="Size of tile to split source image")
    parser.add_argument('--scale', dest='scale', required=False, type=float, default=1.0,
                        help="Images scaling factor")
    parser.add_argument('--skip-empty', dest='skip_empty', required=False, default=False, action='store_true',
                        help="Skip empty labels")
    args = parser.parse_args()

    # Check input directory
    if not os.path.isdir(args.src_root):
        print("Failed to check input '{0}' directory. Abort.".format(args.src_root))
        return -1

    # Convert dataset
    convert_dataset(args.src_root, args.dst_root, args.tile_size, args.scale, args.skip_empty)

    print('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())