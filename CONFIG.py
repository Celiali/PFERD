import os
ProjectPATH = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(ProjectPATH, 'moshpp/src'))
ModelPATH = os.path.join(ProjectPATH, 'hSMALdata/my_smpl_0000_horse_new_skeleton_horse.pkl')
ModelPriorPATH = os.path.join(ProjectPATH, 'hSMALdata/walking_toy_symmetric_smal_0000_new_skeleton_pose_prior_new_36parts.pkl')
ModelNPZPATH = os.path.join(ProjectPATH, 'hSMALdata/my_smpl_0000_horse_new_skeleton_horse.npz')
DatasetPATH = os.path.join(ProjectPATH, 'dataset/DEMO')
DEVICE = "cuda:0"