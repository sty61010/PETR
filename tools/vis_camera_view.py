# matplotlib inline
# from nuscenes.nuscenes import NuScenes
import os
import tqdm
import json
from visual_nuscenes import NuScenes

use_gt = False
model_name = 'depthr_r50dcn_c5_512_1408_gtdepth'
out_dir = f'./result_vis/{model_name}/'
result_json = f"./work_dirs/{model_name}/results_eval/pts_bbox/results_nusc"
dataroot = '/mnt/ssd1/Datasets/nuscenes/v1.0-mini/'
version = 'mini'

#first_sample_token = '747aa46b9a4641fe90db05d97db2acea'
first_sample_token = '3e8750f331d7499e9b5123e9eb70f2e2'

# nusc = NuScenes(version='v1.0-mini', dataroot='/mnt/ssd1/Datasets/nuscenes/v1.0-mini', verbose=True)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if use_gt:
    nusc = NuScenes(version=f'v1.0-{version}', dataroot=dataroot, verbose=True,
                    pred=False, annotations="sample_annotation")
else:
    nusc = NuScenes(version=f'v1.0-{version}', dataroot=dataroot, verbose=True,
                    pred=True, annotations=result_json, score_thr=0.25)


my_sample = nusc.get('sample', first_sample_token)
sensors = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

for sensor in sensors:
    data = nusc.get('sample_data', my_sample['data'][sensor])
    # nusc.render_sample_data(data['token'], with_anns=False, out_path=f'{out_dir}/{sensor}_v.png')
    # nusc.render_sample_data(data['token'], with_anns=True, out_path=f'{out_dir}{sensor}_gt.png')

    nusc.render_sample_data(data['token'], with_anns=True, out_path=f'{out_dir}{sensor}_vpred.png')

    # print(f'file name: {data['filename']}')
    filename = data['filename']
    print(filename)
