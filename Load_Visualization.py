import CONFIG
import numpy as np
import os
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from aitviewer.renderables.spheres import Spheres
from utils.smal import SMALLayer, HSMAL
from utils.readfile import read_mocap,read_results

def find_id(missing, data_length, start, end, downSample):
    enabled_frames = np.ones(data_length, dtype=np.bool8)
    if len(missing) != 0 :
        enabled_frames[missing] = 0
    enabled_frames = enabled_frames[start:end]
    if downSample == 1:
        id_ = np.where(enabled_frames)[0]
        return enabled_frames, id_
    else:
        downSample_flag = np.zeros(enabled_frames.shape, dtype=np.bool8)
        downSample_flag[::downSample] = True
        downSample_flag = np.logical_and(enabled_frames, downSample_flag)
        downSample_enable_frame = downSample_flag[::downSample]
        id_ = np.where(downSample_flag)[0]
        return downSample_enable_frame, id_


def Load_Visualization(ID=1, mocapname='20201128_ID_1_0008', start=None, end = None,downSample = 1, VISUAL_MOCAP = False, num_betas = 10):
    results_path = os.path.join(CONFIG.DatasetPATH, f'ID_{ID}', 'MODEL_DATA', f'{mocapname}_hsmal.npz')
    data = read_results(results_path)
    data_length = data['trans'].shape[0]
    start = 0 if start is None else start
    end = data_length if end is None or end > data_length else end
    betas = data['betas'][start:end,:]
    poses = data['poses'][start:end,:]
    trans = data['trans'][start:end,:]

    missing_frame = data['missing_frame']
    enabled_frames, id_ = find_id(missing_frame, data_length, start, end, downSample = downSample)

    smal_layer = SMALLayer(
        model_path=CONFIG.ModelPATH,
        model_cls=HSMAL,
        device=CONFIG.DEVICE,
        num_betas=num_betas,
    )

    smal_seq = SMPLSequence(
        poses_body=poses[id_,3:],
        smpl_layer=smal_layer,
        poses_root=poses[id_,:3],
        betas=betas[id_],
        trans=trans[id_],
        device=CONFIG.DEVICE,
        color=(149/255, 149/255, 149/255, 0.8),
        z_up=True,
        enabled_frames= enabled_frames
    )

    # Draw an outline around the SMPL mesh.
    smal_seq.mesh_seq.draw_outline = True
    smal_seq.skeleton_seq.enabled = False

    if VISUAL_MOCAP:
        mocapfile = os.path.join(CONFIG.DatasetPATH, f'ID_{ID}', 'C3D_DATA', f'{mocapname}.c3d')
        data,_ = read_mocap(mocapfile)
        mocapdata = data.markers.copy()
        mocapdata = mocapdata[start:end, ...]
        # invisible marker set to nan for not displaying
        mocapdata[mocapdata == 0] = np.nan
        assert mocapdata.shape[0] == poses.shape[0] == trans.shape[0] == betas.shape[0]
        ptc_mocap = Spheres(mocapdata[id_,...],color=(149 / 255, 85 / 255, 149 / 255, 0.5), radius=0.05,rotation=np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
                            enabled_frames=enabled_frames,)
    else:
        ptc_mocap = None

    v = Viewer()
    v.scene.camera.position = (0,3,15)
    v.playback_fps = int(240/downSample)

    v.scene.add(smal_seq)
    if VISUAL_MOCAP:
        v.scene.add(ptc_mocap)
    v.run()

def parse_augment():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ID", type=int, default=1)
    parser.add_argument("--mocapname", type=str, default='20201128_ID_1_0007')
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--downSample", type=int, default=8, help='mocap framerate 240hz, downsample the mocap data')
    parser.add_argument('--VISUAL_MOCAP', action='store_true', help='Whether visualizing')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_augment()
    Load_Visualization(ID=args.ID, mocapname=args.mocapname,start=args.start, end = args.end, downSample = args.downSample, VISUAL_MOCAP = args.VISUAL_MOCAP)
