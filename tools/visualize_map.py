import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def get_class_precision_from_json(json_path: Path, class_name: str) -> np.ndarray:
    with open(json_path, 'r') as f:
        data = json.load(f)

    precisions = []
    for key, class_data in data.items():
        if not key.startswith(class_name):
            continue
        precisions.append(class_data['precision'])

    precisions = np.mean(np.array(precisions), axis=0)
    return precisions


def draw(X: np.ndarray, Ys: Dict[str, np.ndarray], title=None, xlabel=None, ylabel=None):
    for key, Y in Ys.items():
        plt.plot(X, Y, label=key)

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.legend()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('class_name', choices=['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'bicycle', 'traffic_cone', 'barrier'])
    parser.add_argument('json_dirs', nargs='+', type=Path)
    parser.add_argument('-o', '--output-image', type=Path)
    args = parser.parse_args()

    recalls = np.linspace(0.0, 1.0, 101)
    all_models_precisions = {
        json_dir.stem: get_class_precision_from_json(
            json_dir / 'metrics_details.json',
            args.class_name) for json_dir in args.json_dirs}

    assert np.all(np.array([len(precisions) for precisions in all_models_precisions.values()]) == len(recalls))
    draw(recalls, all_models_precisions, title=args.class_name, xlabel='Recall', ylabel='Precision')

    if not args.output_image:
        args.output_image = Path(f'PR_curve_{args.class_name}_{"-".join([json_dir.stem for json_dir in args.json_dirs])}.jpg')
    plt.savefig(args.output_image, bbox_inches='tight')
    print(f"\033[32;1mSaved image to '{args.output_image.relative_to('.')}'\033[0m")
