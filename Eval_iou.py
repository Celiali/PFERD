import CONFIG
import sys
import matplotlib
matplotlib.use('TkAgg')
import cv2, os, torch, glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from utils.readfile import read_results, get_imgs, read_mocap
from utils.project import get_cams_renderers, reproject_masks

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c


def cal_iou( mask_pred, M_gt):
    # function adapted from https://github.com/silviazuffi/smalst
    if np.any(~np.isin(mask_pred, [0, 1])) :
        print('mask_pred contains value(s) other than 0 and 1.')
    if np.any(~np.isin(M_gt, [0, 1])):
        print('M_gt contains value(s) other than 0 and 1.')
    IOU = np.sum(M_gt * mask_pred) / (np.sum(M_gt) + np.sum(mask_pred) - np.sum(M_gt * mask_pred))
    return IOU

def eval_iou(ID=1, mocapname = '20201128_ID_1_0008', VISUAL = True):
    # load hSMAL results
    results_path = os.path.join(CONFIG.DatasetPATH, f'ID_{ID}', 'MODEL_DATA', f'{mocapname}_hsmal.npz')
    resultsdata = read_results(results_path)
    print({k: v if isinstance(v, str) or isinstance(v, float) or isinstance(v, int) else v.shape for k, v in
           resultsdata.items() if not isinstance(v, list) and not isinstance(v, dict)})
    # load mocap data
    mocapfile = os.path.join(CONFIG.DatasetPATH, f'ID_{ID}', 'C3D_DATA', f'{mocapname}.c3d')
    mocapdata,_ = read_mocap(mocapfile)
    mocaplength = mocapdata.markers.shape[0]
    assert mocaplength == resultsdata['poses'].shape[0]

    # load camera and set up render
    campath = os.path.join(CONFIG.DatasetPATH, f'ID_{ID}', 'CAM_DATA')
    cams_list, renderer_list = get_cams_renderers(campath, cameraID=None)

    # search videos
    videos = sorted(glob.glob(os.path.join(CONFIG.DatasetPATH, f'ID_{ID}', 'SEGMENT_DATA', f"{mocapname}_*_seg.mp4")))
    if len(videos) == 0:
        print(f"no seg evaluation selected for {mocapname}")
        sys.exit(0)

    # load model
    bm = BodyModel(bm_fname=CONFIG.ModelNPZPATH, num_betas=10).to(CONFIG.DEVICE)
    faces = c2c(bm.f)

    if VISUAL:
        plt.figure(figsize=(64, 32))
        plt.ion()

    iou_mocap = []
    for t, video_current in enumerate(videos): # one of the camera
        print(video_current)
        vlist = [cv2.VideoCapture(video_current)]
        cams_selected = f"Miqus_Video_{os.path.basename(video_current).split('.')[0].split('_')[-2]}"
        frame_number = [int(v.get(cv2.CAP_PROP_FRAME_COUNT)) for v in vlist]
        fps = [int(v.get(cv2.CAP_PROP_FPS)) for v in vlist]

        interval = int(mocapdata.frame_rate/fps[0])

        FLAG = True
        iou_video = []
        for i in trange(0,frame_number[0]):
            imgs = get_imgs(vlist, i, dis=None) # no need to undistort img
            gt_masks = np.round(imgs[0, :, :, :] / 255.)[:, :, 0]

            frame_selected = i*interval
            if frame_selected >= mocaplength:
                break

            if  (imgs[0, :, :, :] != 0).sum()<1400 or frame_selected in resultsdata['missing_frame'] or (np.where(mocapdata.markers[frame_selected,...] != [0,0,0])[0]).shape[0] < 72:
                FLAG = False
                continue
            else:
                body_parms = {k: torch.Tensor(v[frame_selected,...][None]).to(CONFIG.DEVICE) for k, v in resultsdata.items() if k in ['poses', 'betas', 'trans']}
                body_parms['root_orient'] = body_parms['poses'][:, :3]
                body_parms['pose_body'] = body_parms['poses'][:, 3:]
                body_pose = bm(**body_parms)
                points = body_pose.v[0].cpu().data.numpy()
                proj_masks = reproject_masks(points, faces, renderer_list, cams_list, cameraID = cams_selected, imgs = None, missingframe = False, obtainSil=True)[0]
                FLAG = True
            if FLAG:
                iou = cal_iou(proj_masks, gt_masks)
                iou_video.append(iou)
            if VISUAL and FLAG:
                plt.cla()
                plt.imshow((gt_masks+proj_masks)/2)
                plt.title(f'{mocapname}: frame{i}: iou{iou:.3f}', fontsize=16)
                plt.show()
                plt.pause(0.00001)
        print( f'{mocapname}: {cams_selected}: total frame: {len(iou_video)}: averageIOU: {np.mean(np.array(iou_video))}, stdIOU: {np.std(np.array(iou_video))}, medianIOU: {np.median(np.array(iou_video))}')

        iou_mocap.append(iou_video)
    iou_mocap = np.concatenate(iou_mocap)
    print( f'{mocapname}: total cams {len(videos)}: total frame: {iou_mocap.shape}: averageIOU: {np.mean(iou_mocap)}, stdIOU: {np.std(iou_mocap)}, medianIOU: {np.median(iou_mocap)}')

def parse_augment():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ID", type=int, default=1)
    parser.add_argument("--mocapname", type=str, default='20201128_ID_1_0007')
    parser.add_argument('--VISUAL', action='store_true', help='Whether visualizing')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_augment()
    eval_iou(ID=args.ID, mocapname=args.mocapname, VISUAL=args.VISUAL)