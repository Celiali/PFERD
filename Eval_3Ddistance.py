import CONFIG
import numpy as np
from tqdm import trange
import pickle, os
from psbody.mesh import Mesh,MeshViewer
from psbody.mesh.sphere import Sphere
from psbody.mesh.lines import Lines
from body_visualizer.mesh.psbody_mesh_sphere import points_to_spheres
from moshpp_interface.models_bodymodel_loader import load_moshpp_models
from moshpp_interface.transformed_lm import TransformedCoeffs,TransformedLms
from utils.readfile import read_results, read_mocap

def eval_3ddistance(ID=1, mocapname = '20201128_ID_1_0004', VISUAL = True):
    # load hSMAL results
    results_path = os.path.join(CONFIG.DatasetPATH, f'ID_{ID}', 'MODEL_DATA', f'{mocapname}_hsmal.npz')
    resultsdata = read_results(results_path)
    print({k: v if isinstance(v, str) or isinstance(v, float) or isinstance(v, int) else v.shape for k, v in
           resultsdata.items() if not isinstance(v, list) and not isinstance(v, dict)})

    # load mocap data
    mocapfile = os.path.join(CONFIG.DatasetPATH, f'ID_{ID}', 'C3D_DATA', f'{mocapname}.c3d')
    mocapdata,observed_markers_dict = read_mocap(mocapfile)
    mocaplength = mocapdata.markers.shape[0]
    assert mocaplength == resultsdata['poses'].shape[0]

    # load model
    can_model, opt_models = load_moshpp_models(
        surface_model_fname=CONFIG.ModelPATH,
        surface_model_type='animal_horse',
        optimize_face=False,
        num_beta_shared_models=1,
        pose_body_prior_fname=CONFIG.ModelPriorPATH, )

    # parameterize marker position
    stageI_file =  os.path.join(CONFIG.DatasetPATH, f'ID_{ID}', 'MODEL_DATA', f'ID_{ID}_stagei.npz')
    stageIdata = dict(np.load(stageI_file, allow_pickle=True))
    marker_latent = stageIdata['marker_latent']
    latent_labels = stageIdata['labels']
    can_model.betas[:] = stageIdata['betas'][:].copy()
    tc = TransformedCoeffs(can_body=can_model.r, markers_latent=marker_latent)
    markers_sim_all = TransformedLms(transformed_coeffs=tc, can_body=opt_models[0])
    assert (resultsdata['betas'] == stageIdata['betas'][:10]).all()

    if VISUAL:
        mv = MeshViewer()

    data = []
    for i in trange(mocaplength):
        opt_models[0].pose[:] = resultsdata['poses'][i]
        opt_models[0].trans[:] = resultsdata['trans'][i]

        markers_obs = np.vstack([mocapdata.markers[i][mocapdata.labels.index(l)] if l in mocapdata.labels else np.zeros(3) for l in latent_labels])
        markers_sim = np.vstack([markers_sim_all.r[lid] for lid, l in enumerate(latent_labels)])
        labels = [t for t, l in enumerate(latent_labels) if l not in observed_markers_dict[i]] # these are all zeros since they are invisible mocap
        labels_exists = [t for t, l in enumerate(latent_labels) if l in observed_markers_dict[i]] # visible mocap marker indices
        assert (np.unique(np.where(markers_obs!=0)[0]) == labels_exists).all()

        distances = np.linalg.norm(markers_obs - markers_sim, axis=1)
        if len(labels) != 0:
            distances[labels] = np.nan
        if i in resultsdata['missing_frame']:
            distances[:] = np.nan
        if len(labels_exists) <= 23 :
            distances[:] = np.nan
        data.append(distances)

        if VISUAL:
            body_mesh = Mesh(v=opt_models[0].r, f=opt_models[0].f)

            linev = np.hstack((markers_obs[labels_exists], markers_sim[labels_exists])).reshape((-1, 3))
            linee = np.arange(len(linev)).reshape((-1, 2))
            ll = Lines(v=linev, e=linee)
            ll.vc = (ll.v * 0. + 1) * np.array([0., .5, .6])

            markers_obs_spheres = points_to_spheres(markers_obs[labels_exists], point_color=np.array((0., 1., 0.)), radius=0.009)  # green
            markers_sim_spheres = points_to_spheres(markers_sim_all[labels_exists].r, point_color=np.array((1., 0., 0.)), radius=0.009)  # red

            mv.set_dynamic_meshes([markers_obs_spheres, markers_sim_spheres, body_mesh])
            mv.set_dynamic_lines([ll])

    print('nanmean',np.nanmean(data))
    print('nanmedian',np.nanmedian(data))
    nanmax_index = np.where(data == np.nanmax(data))
    print('nanmax', np.nanmax(data), 'nanmax_markers', latent_labels[nanmax_index[1][0]])
    nanmin_index = np.where(data == np.nanmin(data))
    print('nanmin', np.nanmin(data), 'nanmin_markers', latent_labels[nanmin_index[1][0]])
    marker_each = np.nanmean(data, axis=0)
    markers_not_observed_in_stageI = [stageIdata['labels'][~stageIdata['flag'].astype(bool)], marker_each[~stageIdata['flag'].astype(bool)]]
    print(markers_not_observed_in_stageI)
    print(marker_each)

def parse_augment():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ID", type=int, default=1)
    parser.add_argument("--mocapname", type=str, default='20201128_ID_1_0004')
    parser.add_argument('--VISUAL', action='store_true', help='Whether visualizing')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_augment()
    eval_3ddistance(ID=args.ID, mocapname=args.mocapname, VISUAL=True)#args.VISUAL)