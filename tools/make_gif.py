"""
Usage 1:
    Generate the camera gif.
    python tools/make_dif.py {IMAGE_DIR} [--add-text] [-o output.gif]
Usage 2:
    Generate the lidar gif.
    python tools/make_dif.py {IMAGE_DIR} --sensors LIDAR_TOP [-o lidar.gif] [--add-text]
"""
import argparse
import os
from pathlib import Path

import cv2
import imageio
import numpy as np

SENSORS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'LIDAR_TOP']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=Path)
    # parser.add_argument('--sensors', choices=SENSORS, nargs='+',
    #                     default=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'])
    parser.add_argument('--sensors', choices=SENSORS, nargs='+',
                        default=['CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'])
    # parser.add_argument('--sensors', choices=SENSORS, nargs='+',
    #                     default=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'])
    parser.add_argument('-o', '--output-gif', type=Path)
    parser.add_argument('-d', '--duration', type=int, default=200,
                        help='Durations for each frame in milliseconds.')
    parser.add_argument('--add-text', action='store_true')
    args = parser.parse_args()

    assert 'LIDAR_TOP' not in args.sensors or len(args.sensors) == 1
    if args.output_gif is None:
        args.output_gif = f'{args.image_dir.stem}.gif'

    image_subdirs = sorted(args.image_dir.glob('*'), key=lambda x: int(x.stem))
    all_frame_images = []
    for i, subdir in enumerate(image_subdirs):
        if 'LIDAR_TOP' in args.sensors:
            image = cv2.imread(os.path.join(subdir, 'LIDAR_TOP_pred.png'))
        else:
            images = [cv2.imread(os.path.join(subdir, f'{camera}_pred.png')) for camera in args.sensors]
            images = np.array(images).reshape((-1, 3, *images[0].shape))
            images = [np.concatenate(image_row, axis=1) for image_row in images]
            image = np.concatenate(images, axis=0)

        if args.add_text:
            image = cv2.putText(image, str(i), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        all_frame_images.append(image)

    imageio.mimsave(args.output_gif, all_frame_images, duration=args.duration / 1000)
