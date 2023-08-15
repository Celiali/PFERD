import CONFIG
import torch, cv2
import numpy as np
from utils.readfile import read_cams

def get_cams_renderers(cam_path, cameraID = None):
    from utils.render import Renderer
    cams = read_cams(cam_path)
    cams_keys = [i for i in cams.keys() if cameraID in i] if cameraID is not None else sorted(list(cams.keys()))
    assert len(cams_keys) > 0
    renderer_list = {}
    cams_list = {}
    for key in cams_keys:
        renderer = Renderer(focal_length_x=cams[key]['focal'][0], focal_length_y=cams[key]['focal'][1],
                            img_w=cams[key]['resolution'][0],  img_h=cams[key]['resolution'][1],
                            same_mesh_color=False, camera_center= [cams[key]['K'][0,2], cams[key]['K'][1,2]])
        renderer_list[key] = renderer
        cams_list[key] = cams[key]
    return cams_list, renderer_list


def reproject_masks(vertex, faces, renderer_list, cams_list, cameraID = None, imgs = None, missingframe = False, obtainSil=False):
    '''
    Code adapted from https://github.com/marcbadger/avian-mesh
    Args:
        imgs: color image as input for background
        missing: if the frame is missing, return gray meshes
    Returns:
    '''
    # Transform vertex for each camera view
    cams_keys = [i for i in cams_list.keys() if cameraID in i] if cameraID is not None else sorted(list(cams_list.keys()))
    rotation = torch.Tensor([cams_list[i]['R'] for i in cams_keys]).float().to(CONFIG.DEVICE) #(1,3,3)
    translation = torch.Tensor([cams_list[i]['T'] for i in cams_keys]).float().to(CONFIG.DEVICE) #(1,3)

    vertex_torch = torch.Tensor(vertex).float().to(CONFIG.DEVICE).unsqueeze(0)
    points = vertex_torch.repeat([len(cams_keys), 1, 1])
    points = torch.einsum('bij,bkj->bki', rotation, points) + translation

    # Apply Distortion
    if 'D' in list(cams_list[cams_keys[0]].keys()):
        kc = [cams_list[i]['D'] for i in cams_keys]
        kc = torch.tensor(np.array(kc)).float().to(CONFIG.DEVICE)
        d = points[:, :, 2:]
        points = points[:, :, :] / points[:, :, 2:]

        r2 = points[:, :, 0] ** 2 + points[:, :, 1] ** 2
        dx = (2 * kc[:, [2]] * points[:, :, 0] * points[:, :, 1]
              + kc[:, [3]] * (r2 + 2 * points[:, :, 0] ** 2))

        dy = (2 * kc[:, [3]] * points[:, :, 0] * points[:, :, 1]
              + kc[:, [2]] * (r2 + 2 * points[:, :, 1] ** 2))

        x = (1 + kc[:, [0]] * r2 + kc[:, [1]] * r2.pow(2) + kc[:, [4]] * r2.pow(3)) * points[:, :, 0] + dx
        y = (1 + kc[:, [0]] * r2 + kc[:, [1]] * r2.pow(2) + kc[:, [4]] * r2.pow(3)) * points[:, :, 1] + dy

        points = torch.stack([x, y, torch.ones_like(x)], dim=-1) * d

    proj_masks = []
    for i in range(len(cams_keys)):
        # Render for each view
        if imgs is None:
            img = None
        else:
            img= imgs[i, :, :, ::-1].copy()
        renderer = renderer_list[cams_keys[i]]
        results = renderer.render(verts =points[i].cpu().numpy(),faces = faces,
                                       cam_rot = np.eye(3), cam_t = [0, 0, 0], bg_img_rgb = img,
                                       missingframe = missingframe)

        if obtainSil:
            proj_masks.append(results['mask'])
        elif imgs is None:
            proj_masks.append(results['color_rgb'])
        else:
            proj_masks.append(results['bg_img_rgb'])
    return proj_masks

def reproject_keypoints(mesh_keypoints, cams_list, cameraID = None):
    cams_keys = [i for i in cams_list.keys() if cameraID in i] if cameraID is not None else sorted(list(cams_list.keys()))
    proj_kpts_numpy = []
    for i in range(len(cams_keys)):
        key = cams_keys[i]
        # Convert to a rotation vector
        rvec, _ = cv2.Rodrigues(cams_list[key]['R'])
        projected_points, _ = cv2.projectPoints(mesh_keypoints, rvec, cams_list[key]['T'], cams_list[key]['K'],cams_list[key]['D'])
        proj_kpts_numpy.append(projected_points.squeeze(1))
    proj_kpts_numpy = np.array(proj_kpts_numpy)
    return proj_kpts_numpy