import os
import trimesh
import pyrender
import numpy as np
import colorsys

class Renderer(object):
    """
    Code adapted from https://github.com/haofanwang/CLIFF
    """
    def __init__(self, focal_length_x=600, focal_length_y = None, img_w=512, img_h=512,same_mesh_color=False, camera_center = None):
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w,
                                                   viewport_height=img_h,
                                                   point_size=1.0)
        if camera_center is None:
            self.camera_center = [img_w // 2, img_h // 2]
        else:
            self.camera_center = camera_center
        if focal_length_y == None:
            self.focal_length_x = focal_length_x
            self.focal_length_y = focal_length_x
        else:
            self.focal_length_x = focal_length_x
            self.focal_length_y = focal_length_y
        self.same_mesh_color = same_mesh_color
        self.right_index = range(813, 1497)
        self.left_index = range(0, 813)
        self.right_faces = range(1, 2991, 2)
        self.left_faces = range(0, 2990, 2)

    def render(self, verts, faces, bg_img_rgb=None, bg_color=(0, 0, 0, 0),cam_rot = None, cam_t = None, missingframe= False):
        '''
        Args:
            verts: [N, 3]
            bg_img_rgb: background img
            missingframe: flag for missing frame
        Returns:
        '''
        # Create a scene and camera
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=np.ones(3) * 0)
        camera_pose = np.eye(4)
        if (cam_rot != None).all() and cam_t != None:
            camera_pose[:3, :3] = cam_rot
            camera_pose[:3, 3] = cam_t
        camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length_x, fy=self.focal_length_y,
                                                  cx=self.camera_center[0], cy=self.camera_center[1]) #zfar=1000)
        scene.add(camera, pose=camera_pose)

        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        light_pose = trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0])
        scene.add(light, pose=light_pose)
        light_pose = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])
        scene.add(light, pose=light_pose)

        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])

        mesh = trimesh.Trimesh(verts, faces)
        mesh.apply_transform(rot)
        if self.same_mesh_color:
            mesh_color = colorsys.hsv_to_rgb(0.6, 0.5, 1.0)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=mesh_color)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material, wireframe=False)
        else:
            colors_faces = np.zeros_like(mesh.faces)
            if missingframe:
                colors_faces[self.left_faces, :] =np.array([0.08, 0.08,0.08])*255.
                colors_faces[self.right_faces, :] = np.array([0.09,0.09,0.09])*255.
            else:
                colors_faces[self.left_faces, :] = np.array([0.654, 0.396, 0.164])*255.
                colors_faces[self.right_faces, :] = np.array([.7, .7, .7])*255.
            mesh.visual.face_colors = colors_faces  # np.random.uniform(size=mesh.faces.shape)
            mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene.add(mesh, 'mesh')

        color_rgba, depth_map = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color_rgb = color_rgba[:, :, :3]
        mask = depth_map > 0
        if bg_img_rgb is not None:
            bg_img_rgb[mask] = color_rgb[mask] * 0.7 + bg_img_rgb[mask] * 0.3
        return {'mask':mask.astype(float), 'color_rgb': color_rgb, 'bg_img_rgb': bg_img_rgb}

    def delete(self):
        """
        Need to delete before creating the renderer next time
        """
        self.renderer.delete()