import CONFIG
import os
import numpy as np
import glob
import cv2
from moshpp.tools.mocap_interface import MocapSession

def read_mocap(mocap_fname):
    mocap = MocapSession(mocap_fname=mocap_fname,
                         mocap_unit='mm',
                         mocap_rotate=None,
                         labels_map=None,
                         ignore_stared_labels=False)
    observed_markers_dict = mocap.markers_asdict()
    return mocap, observed_markers_dict

def read_results(results_file):
    data = np.load(results_file, allow_pickle=True)
    output = dict(data)
    T = data['trans'].shape[0]
    output['betas'] = data['betas'][None].repeat(T, axis=0)
    return output

def get_imgs(vlist, frameID, dis = None):
    '''
    Args:
        dis: used for undistortion
    '''
    imgs = np.zeros([len(vlist), 1088, 1920, 3])
    for i in range(len(vlist)):
        vlist[i].set(cv2.CAP_PROP_POS_FRAMES, frameID)
        flag, img = vlist[i].read()
        if not flag:
            img = np.zeros((1088,1920,3), dtype='uint8')
        if dis is not None:
            img = cv2.undistort(img, dis[i][0], dis[i][1])
        imgs[i, ...] = img
    return imgs

def read_cams(cam_path):
    cams_file = sorted(glob.glob(os.path.join(cam_path, '*.npz')))
    cams = {}
    for i in cams_file:
        cams_name = os.path.basename(i).split('.')[0][7:]
        data = np.load(i, allow_pickle=True)
        rotation = data['R']
        translation = data['T']
        instrinsic = data['K']
        dis = data['D']
        focal = np.array([instrinsic[0,0], instrinsic[1,1]])
        center = np.array([instrinsic[0,2], instrinsic[1,2]])
        resolution = [1920, 1088]
        cams[cams_name] = {'R': rotation, 'T': translation, 'K': instrinsic, 'D': dis,
                           'focal':focal, 'center': center, 'resolution':resolution}
    return cams


if __name__ == '__main__':
    mocapfilename = os.path.join(CONFIG.DatasetPATH, 'ID_1/C3D_DATA/20201128_ID_1_0004.c3d')
    data = read_mocap(mocapfilename)
    print("data")
    results_path = os.path.join(CONFIG.DatasetPATH, 'ID_1/MODEL_DATA/20201128_ID_1_0004_hsmal.npz')
    results = read_results(results_path)
    print({k: v if isinstance(v, str) or isinstance(v, float) or isinstance(v, int) else v.shape for k, v in
           results.items() if not isinstance(v, list) and not isinstance(v, dict)})
