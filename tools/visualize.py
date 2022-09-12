import os
import tqdm
import json
from visual_nuscenes import NuScenes
use_gt = False
model_name = 'depthr_r50dcn_c5_512_1408_gtdepth'
# model_name = 'petr_r50dcn_gridmask_c5'
# model_name = 'petr_vovnet_gridmask_p4_512_1408_bs2'


out_dir = f'./result_vis/{model_name}/'
result_json = f"./work_dirs/{model_name}/results_eval/pts_bbox/results_nusc"
dataroot = '/mnt/ssd1/Datasets/nuscenes/v1.0-mini/'
version = 'mini'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if use_gt:
    nusc = NuScenes(version=f'v1.0-{version}', dataroot=dataroot, verbose=True,
                    pred=False, annotations="sample_annotation")
else:
    nusc = NuScenes(version=f'v1.0-{version}', dataroot=dataroot, verbose=True,
                    pred=True, annotations=result_json, score_thr=0.25)

with open('{}.json'.format(result_json)) as f:
    table = json.load(f)
tokens = list(table['results'].keys())

for token in tqdm.tqdm(tokens[:100]):
    if use_gt:
        nusc.render_sample(token, out_path=f"{out_dir}{token}_gt.png", verbose=False)
    else:
        nusc.render_sample(token, out_path=f"{out_dir}{token}_pred.png", verbose=False)
