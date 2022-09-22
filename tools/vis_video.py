# matplotlib inline
# from nuscenes.nuscenes import NuScenes
import os
import tqdm
import json
from visual_nuscenes import NuScenes

use_gt = True
# model_name = 'depthr_r50dcn_c5_512_1408_gtdepth'
# model_name = 'petr_r50dcn_gridmask_c5'
model_name = 'petr_vovnet_gridmask_p4_512_1408_bs2'
# model_name = 'work_dirs_bs8/depthr_r101cn_p4_512_1408_depth32_ddn16_w10_lid64_start1e-3_en3_de3_view_dsv_bs2'
# model_name = 'depthr_r50dcn_p4_512_1408_depth32_ddn16_w10_lid64_start1e-3_en3_de3_view_dsv_cbgs_bs2'

# dir_name = 'ours'
# dir_name = 'baseline'
# dir_name = 'baseline_pretrained'
dir_name = 'demo_video/gt'


# out_dir = f'./result_vis/{model_name}/'
out_dir = f'/home/cytseng/qualitative/{dir_name}/'

result_json = f"./work_dirs/{model_name}/results_eval/pts_bbox/results_nusc"
dataroot = '/mnt/ssd1/Datasets/nuscenes/v1.0-mini/'
version = 'mini'

first_sample_token = '747aa46b9a4641fe90db05d97db2acea'
# first_sample_token = '3e8750f331d7499e9b5123e9eb70f2e2'
tokens = ['3e8750f331d7499e9b5123e9eb70f2e2', '3950bd41f74548429c0f7700ff3d8269', 'c5f58c19249d4137ae063b0e9ecd8b8e', '700c1a25559b4433be532de3475e58a9', '747aa46b9a4641fe90db05d97db2acea',
          'f4f86af4da3b49e79497deda5c5f223a', '6832e717621341568c759151b5974512', 'c59e60042a8643e899008da2e446acca', 'fa65a298c01f44e7a182bbf9e5fe3697', 'a98fba72bde9433fb882032d18aedb2e', 'b6b0d9f2f2e14a3aaa2c8aedeb1edb69', '796b1988dd254f74bf2fb19ba9c5b8c6', '0d0700a2284e477db876c3ee1d864668']

# tokens = ['3e8750f331d7499e9b5123e9eb70f2e2', '3950bd41f74548429c0f7700ff3d8269', 'c5f58c19249d4137ae063b0e9ecd8b8e', '700c1a25559b4433be532de3475e58a9', '747aa46b9a4641fe90db05d97db2acea',
#   'f4f86af4da3b49e79497deda5c5f223a', '6832e717621341568c759151b5974512', 'c59e60042a8643e899008da2e446acca', 'fa65a298c01f44e7a182bbf9e5fe3697', 'a98fba72bde9433fb882032d18aedb2e', 'b6b0d9f2f2e14a3aaa2c8aedeb1edb69', '796b1988dd254f74bf2fb19ba9c5b8c6', '0d0700a2284e477db876c3ee1d864668']
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
# sensors = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
# sensors = ['LIDAR_TOP']
sensors = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'LIDAR_TOP']


count = 0
for token in tokens:
    print(f'token: {token}')
    for sensor in sensors:
        frame_dir = f'{out_dir}{count}/'
        if not os.path.exists(frame_dir):
            os.mkdir(frame_dir)

        my_sample = nusc.get('sample', token=token)

        data = nusc.get('sample_data', my_sample['data'][sensor])
        # nusc.render_sample_data(data['token'], with_anns=False, out_path=f'{out_dir}/{sensor}_v.png')
        # nusc.render_sample_data(data['token'], with_anns=True, out_path=f'{out_dir}{sensor}_gt.png')

        # nusc.render_sample_data(data['token'], with_anns=True, nsweeps=5, out_path=f'{out_dir}{sensor}_bev_green.png')
        nusc.render_sample_data(data['token'], verbose=False, with_anns=True, out_path=f'{frame_dir}{sensor}_pred.png')

        # print(f'file name: {data['filename']}')
        filename = data['filename']
        print(filename)
    count += 1
