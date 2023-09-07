import CONFIG
import matplotlib
matplotlib.use('TkAgg')
import cv2, os, torch, glob
import numpy as np
import matplotlib.pyplot as plt
from utils.readfile import read_results, get_imgs, read_mocap
from utils.project import get_cams_renderers, reproject_masks, reproject_keypoints

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

def projection(ID=1, mocapname = '20201128_ID_1_0008', cameraID = None, start=None, end = None, VISUAL = True, VISUAL_MOCAP = True):
    # load hSMAL results
    results_path = os.path.join(CONFIG.DatasetPATH, f'ID_{ID}', 'MODEL_DATA', f'{mocapname}_hsmal.npz')
    resultsdata = read_results(results_path)
    print({k: v if isinstance(v, str) or isinstance(v, float) or isinstance(v, int) else v.shape for k, v in
           resultsdata.items() if not isinstance(v, list) and not isinstance(v, dict)})
    # load mocap data
    mocapfile = os.path.join(CONFIG.DatasetPATH, f'ID_{ID}', 'C3D_DATA', f'{mocapname}.c3d')
    mocapdata,_ = read_mocap(mocapfile)
    selected_marker_labels_left = [i_ml for i_ml, ml in enumerate(mocapdata.labels) if 'L_' in ml or 'LF_' in ml or 'LH_' in ml]
    selected_marker_labels_right = [i_ml for i_ml, ml in enumerate(mocapdata.labels) if 'R_' in ml or 'RF_' in ml or 'RH_' in ml]
    selected_marker_labels_middle = [i_ml for i_ml, ml in enumerate(mocapdata.labels) if 'L_' not in ml and 'LF_' not in ml and 'LH_' not in ml and 'R_' not in ml and 'RF_' not in ml and 'RH_' not in ml]
    assert mocapdata.markers.shape[0] == resultsdata['poses'].shape[0]

    # load camera and set up render
    campath = os.path.join(CONFIG.DatasetPATH, f'ID_{ID}', 'CAM_DATA')
    cams_list, renderer_list = get_cams_renderers(campath, cameraID=cameraID)

    # load videos
    videofolder = os.path.join(CONFIG.DatasetPATH, f'ID_{ID}', 'VIDEO_DATA', mocapname)
    videos = [glob.glob(os.path.join(videofolder, f"{mocapname}_*{cam.split('_')[-1]}.avi"))[0] for cam in sorted(list(cams_list.keys()))]
    assert len(videos) == len(cams_list.keys())
    vlist = [cv2.VideoCapture(video_current) for video_current in videos]
    frame_number = [int(v.get(cv2.CAP_PROP_FRAME_COUNT)) for v in vlist]
    fps = [int(v.get(cv2.CAP_PROP_FPS)) for v in vlist]
    assert len(set(fps)) == 1
    # load model
    bm = BodyModel(bm_fname=CONFIG.ModelNPZPATH, num_betas=10).to(CONFIG.DEVICE)
    faces = c2c(bm.f)

    interval = int(mocapdata.frame_rate/fps[0])

    start = 0 if start is None else start
    end = min(frame_number) if end is None or end > min(frame_number) else end

    if VISUAL:
        plt.figure(figsize=(64, 32))
        plt.ion()

    for i in range(start, end):
        body_parms = {k: torch.Tensor(v[i*interval,...][None]).to(CONFIG.DEVICE) for k, v in resultsdata.items() if k in ['poses', 'betas', 'trans']}
        body_parms['root_orient'] =  body_parms['poses'][:,:3]
        body_parms['pose_body'] = body_parms['poses'][:,3:]
        body_pose = bm(**body_parms)
        points = body_pose.v[0].cpu().data.numpy()

        imgs = get_imgs(vlist, i , dis = None)
        missingframe = True if i*interval in resultsdata['missing_frame'] else False
        if missingframe:
            proj_masks = [imgs[i_t, :, :, ::-1] for i_t in range(imgs.shape[0])]  # = imgs[i, :, :, ::-1].copy()
        else:
            proj_masks = reproject_masks(vertex = points, faces = faces, renderer_list = renderer_list, cams_list = cams_list, cameraID = cameraID, imgs = imgs, obtainSil=False)
            if VISUAL_MOCAP:
                marker3d = mocapdata.markers[i*interval,...]
                proj_kpts = reproject_keypoints(marker3d, cams_list, cameraID = cameraID)

            for index in range(len(vlist)):
                if VISUAL_MOCAP:
                    for t in range(proj_kpts.shape[1]):
                        if t in selected_marker_labels_left:
                            color = (255,0,0)
                        elif t in selected_marker_labels_right:
                            color = (0,0,255)
                        elif t in selected_marker_labels_middle:
                            color = (0,255,0)
                        cv2.circle(proj_masks[index], (int(proj_kpts[index, t, 0]), int(proj_kpts[index, t, 1])), radius=3,color=color, thickness=-1)
                cv2.putText(proj_masks[index], sorted(list(cams_list.keys()))[index], (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2.,(255, 0, 0), 3)

        if cameraID is not None:
            imgs_all =  proj_masks[0]
        else:
            imgs_all = np.vstack([np.hstack([proj_masks[i] for i in range(5)]), np.hstack([proj_masks[i] for i in range(5,10)])])

        # for visualize
        if VISUAL:
            plt.imshow(imgs_all.astype(np.uint8))
            plt.title(f'{mocapname}: frame{i}', fontsize=16)
            plt.show()
            plt.pause(0.00001)

def parse_augment():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ID", type=int, default=1)
    parser.add_argument("--mocapname", type=str, default='20201128_ID_1_0008')
    parser.add_argument("--cameraID", type=str, default=None, help='None, 20715, 21386, 23348, 23350, 23414, 23415, 23416, 23417, 23603, 23604')
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument('--VISUAL', action='store_true', help='Visualize model')
    parser.add_argument('--VISUAL_MOCAP', action='store_true', help='Visualize mocap')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_augment()
    projection(ID =args.ID, mocapname = args.mocapname, cameraID = args.cameraID, start= args.start, end = args.end, VISUAL = args.VISUAL, VISUAL_MOCAP = args.VISUAL_MOCAP)

