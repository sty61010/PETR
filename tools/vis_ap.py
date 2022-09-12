import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def get_class_precision_from_json(json_path: Path, class_name: str, json_dir_nams: str) -> Dict[str, np.ndarray]:
    with open(json_path, 'r') as f:
        data = json.load(f)

    precisions = []
    precisions_dist = dict()
    for key, class_data in data.items():
        if not key.startswith(class_name):
            continue
        if key[-3:] == '2.0':
            continue

        print(f'{json_dir_nams} with Dist. : ({key[-3:]})')
        precisions.append(class_data['precision'])
        precisions_dist[f'{json_dir_nams} with Dist. : ({key[-3:]})'] = class_data['precision']
    # print(f'precisions: {precisions}')
    precisions = np.mean(np.array(precisions), axis=0)
    return precisions_dist


def draw(X: np.ndarray, Ys: Dict[str, np.ndarray], title=None, xlabel=None, ylabel=None):
    for key, Y in Ys.items():
        print(f'dist: {key[-4:-1]}')
        dist = key[-4:-1]
        if dist == '0.5':
            color = 'red'
        elif dist == '1.0':
            color = 'blue'
        elif dist == '2.0':
            color = 'red'
        else:
            color = 'green'

        if not key.startswith('Baseline'):
            linestyle = "-"
        else:
            linestyle = "--"
        # plt.plot(X, Y, color=color, linestyle=linestyle, marker="s")
        if dist == '1.0':
            plt.plot(X, Y, label=key[:-19], color=color, linestyle=linestyle)
        else:
            plt.plot(X, Y, color=color, linestyle=linestyle)

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel, fontsize=16)
    if ylabel:
        plt.ylabel(ylabel, fontsize=16)
    # plt.legend(labels=["Ours", "PETR", "Dist : (0.5)",  "Dist : (1.0)",  "Dist : (4.0)"], )
    # plt.legend(labels=["Ours", "PETR"], linestyles=["-", "--"])
    # plt.legend()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('class_name', choices=['car', 'truck', 'bus', 'trailer',
                        'construction_vehicle', 'pedestrian', 'bicycle', 'traffic_cone', 'barrier'])
    parser.add_argument('json_dirs', nargs='+', type=Path)
    parser.add_argument('-o', '--output-image', type=Path)
    args = parser.parse_args()

    recalls = np.linspace(0.0, 1.0, 101)
    # all_models_precisions = {
    #     json_dir.stem: get_class_precision_from_json(
    #         json_dir / 'metrics_details.json',
    #         args.class_name,
    #         json_dir.stem,
    #         ) for json_dir in args.json_dirs}
    all_models_precisions = dict()
    for json_dir in args.json_dirs:
        all_models_precisions.update(get_class_precision_from_json(
            json_dir / 'metrics_details.json',
            args.class_name,
            json_dir.stem,
        ))
    print(all_models_precisions)

    assert np.all(np.array([len(precisions) for precisions in all_models_precisions.values()]) == len(recalls))
    # draw(recalls, all_models_precisions, title=args.class_name, xlabel='Recall', ylabel='Precision')
    draw(recalls, all_models_precisions, xlabel='Recall', ylabel='Precision')

    if not args.output_image:
        args.output_image = Path(
            f'PR_curve_{args.class_name}_{"-".join([json_dir.stem for json_dir in args.json_dirs])}.svg')
    plt.savefig(args.output_image, bbox_inches='tight', format='svg')
    # plt.savefig(args.output_image, bbox_inches='tight')

    print(f"\033[32;1mSaved image to '{args.output_image.relative_to('.')}'\033[0m")
