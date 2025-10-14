#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:43:23 2023

@author: huzhe
"""

import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm
from matplotlib import pyplot as plt

from .ray_utils import get_ray_directions_blender, get_rays, read_pfm, data_preproc, data_reg, touint8

blender2opencv = torch.Tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

def translation_mat(rot_matrix, theta):
    theta = np.deg2rad(theta)
    x_vec = rot_matrix * np.sin(theta)*2
    y_vec = 0
    z_vec = rot_matrix * np.cos(theta)*2
    trans_mat = np.array([[x_vec], [y_vec], [z_vec]])
    return trans_mat

def rot_x(theta):
    theta = np.deg2rad(theta)
    rot_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )
    return rot_matrix


def rot_y(theta):
    theta = np.deg2rad(theta)
    rot_matrix = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    return rot_matrix


def rot_z(theta):
    theta = np.deg2rad(theta)
    rot_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return rot_matrix


def pose_spherical(theta_x, theta_y, theta_z):
    rot_mat = rot_z(theta_z) @ (rot_y(theta_y) @ rot_x(theta_x))# rotation matrices multiply
    trans_mat = translation_mat(1., theta_y) # translation matrix
    pad_mat = np.array([0., 0., 0., 1.]).reshape(1, 4)
    world_mat = np.concatenate((rot_mat, trans_mat), 1)
    world_mat = np.concatenate((world_mat, pad_mat), 0)
    return world_mat


class XMPIDataset(Dataset):
    def __init__(
        self,
        datadir,
        split="train",
        downsample=1.0,
        is_stack=False,
        cal_fine_bbox=True,
        N_vis=-1,
        time_scale=1.0,
        scene_bbox_min=[-1.0, -1.0, -1.0],
        scene_bbox_max=[1.0, 1.0, 1.0],
        N_random_pose=1000,
    ):
        self.dtpye = torch.float
        self.root_dir = datadir
        self.split = split
        self.downsample = downsample
        self.img_wh = (int(128), int(128))
        self.is_stack = is_stack
        self.N_vis = N_vis  # evaluate images for every N_vis images

        self.time_scale = time_scale
        self.world_bound_scale = 1.1

        self.near = 0.
        self.far = 4
        self.near_far = [0, 4]

        self.scene_bbox = torch.FloatTensor([scene_bbox_min, scene_bbox_max])
        
        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        self.read_meta()  # Read meta data
        # Calculate a more fine bbox based on near and far values of each ray.
        if cal_fine_bbox:
            if split == 'train':
                xyz_min, xyz_max = self.compute_bbox()
                self.scene_bbox = torch.stack((xyz_min, xyz_max), dim=0)
        self.white_bg = False
        self.ndc_ray = False
        self.depth_data = False

        self.N_random_pose = N_random_pose

        # Generate N_random_pose random poses, which we could render depths from these poses and apply depth smooth loss to the rendered depth.
        if split == "train":
            self.init_random_pose()

    def init_random_pose(self):
        # Randomly sample N_random_pose radius, phi, theta and times.
        theta_x = 0
        theta_y = np.random.rand(self.N_random_pose) * 180
        theta_z = 0
        random_times = self.time_scale * (torch.rand(self.N_random_pose) * 1)
        self.random_times = random_times

        # Generate rays from random radius, phi, theta and times.
        self.random_rays = []
        for i in range(self.N_random_pose):
            random_poses = pose_spherical(theta_x, theta_y[i], theta_z)
            random_poses = torch.from_numpy(random_poses).float()
            rays_o, rays_d = get_rays(self.img_wh[0], self.img_wh[1],random_poses)
            self.random_rays += [torch.cat([rays_o, rays_d], 1)]

        self.random_rays = torch.stack(self.random_rays, 0).reshape(
            -1, *self.img_wh[::-1], 6
        )

    def compute_bbox(self):
        print("compute_bbox_by_cam_frustrm: start")
        xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
        xyz_max = -xyz_min
        rays_o = self.all_rays[:, 0:3]
        viewdirs = self.all_rays[:, 3:6]
        pts_nf = torch.stack(
            [rays_o + viewdirs * self.near, rays_o + viewdirs * self.far]
        )
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0, 1)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0, 1)))
        print("compute_bbox_by_cam_frustrm: xyz_min", xyz_min)
        print("compute_bbox_by_cam_frustrm: xyz_max", xyz_max)
        print("compute_bbox_by_cam_frustrm: finish")
        xyz_shift = (xyz_max - xyz_min) * (self.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
        return xyz_min, xyz_max

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth

    def read_meta(self):
        #with open(os.path.join(self.root_dir, f"transforms_{self.split}.json")) as f:
            #self.meta = json.load(f)

        w, h = self.img_wh
        self.poses = []
        self.all_rays = []
        self.all_times = []
        self.all_imgs = []
        self.all_depth = []
        self.idx = 1

        images = np.load('/data/staff/tomograms/users/zhehu/Hexplane/dataset/sphere/sphere.npy')
        self.imgs = torch.from_numpy(images)

        ProjAngles = np.linspace(0, 180., 64, endpoint=False)
        theta_x = 0
        theta_z = 0
        theta_y = ProjAngles
        if self.split != 'train':
            theta_y += 180
        world_mat = []
        for i in range(len(theta_y)):
            world_mat.append(pose_spherical(theta_x, theta_y[i], theta_z))
        world_matrix = np.asarray(world_mat).astype(float)
        self.imgs_poses = torch.from_numpy(world_matrix).float()

        self.timestemp = torch.arange(self.imgs.shape[0])*0.01-1

        img_eval_interval = (
            1 if self.N_vis < 0 else self.imgs[0] // self.N_vis
        )
        idxs = list(range(0, self.imgs.shape[0], img_eval_interval))
        for i in tqdm(
            idxs, desc=f"Loading data {self.split} ({len(idxs)})"
        ):  # img_list:#
            img = self.imgs[i]
            img = img.view(1, -1).permute(1, 0)  # (h*w, 1) 
            self.all_imgs += [img]
            
            pose = self.imgs_poses[i]
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]
            rays_o, rays_d = get_rays(self.img_wh[0], self.img_wh[1], c2w)  # Get rays, both (h*w, 3).          
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
            #np.save(f'/gpfs/offline1/staff/tomograms/users/zhehu/Hexplane/ray_o/rays_o{i}.npy',rays_o)
            #np.save(f'/gpfs/offline1/staff/tomograms/users/zhehu/Hexplane/ray_o/rays_d{i}.npy',rays_d)
            #cur_time = torch.tensor(
            #    frame["time"]
            #    if "time" in frame
            #    else float(i) / (len(self.meta["frames"]) - 1)
            #).expand(rays_o.shape[0], 1)
            cur_time = self.timestemp[i].expand(rays_o.shape[0], 1)
            self.all_times += [cur_time]

        self.poses = torch.stack(self.poses)
        #  self.is_stack stacks all images into a big chunk, with shape (N, H, W, 3).
        #  Otherwise, all images are kept as a set of rays with shape (N_s, 3), where N_s = H * W * N
        if not self.is_stack:
            self.all_rays = torch.cat(
                self.all_rays, 0
            )  # (len(self.meta['frames])*h*w, 6)
            self.all_imgs = torch.cat(
                self.all_imgs, 0
            )  # (len(self.meta['frames])*h*w, 1)
            self.all_times = torch.cat(self.all_times, 0)

        else:
            self.all_rays = torch.stack(
                self.all_rays, 0
            )  # (len(self.meta['frames]),h*w, 3)
            self.all_imgs = torch.stack(self.all_imgs, 0).reshape(
                -1, *self.img_wh[::-1], 1
            )  # (len(self.meta['frames]),h,w,3)
            self.all_times = torch.stack(self.all_times, 0)

        self.all_times = self.time_scale * (self.all_times * 2.0+1)

    def __len__(self):
        return len(self.all_imgs)

    def get_val_pose(self):
        """
        Get validation poses and times (NeRF-like rotating cameras poses).
        """
        p = [torch.FloatTensor(pose_spherical(0 ,angle, 0)) for angle in np.linspace(0, 180, 40 + 1)[:-1]]
        render_poses = torch.stack(p,0)
        render_times = torch.linspace(0.0, 40.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, self.time_scale * render_times

    def get_val_rays(self):
        """
        Get validation rays and times (NeRF-like rotating cameras poses).
        """
        val_poses, val_times = self.get_val_pose()  # get valitdation poses and times
        rays_all = []  # initialize list to store [rays_o, rays_d]

        for i in range(val_poses.shape[0]):
            c2w = torch.FloatTensor(val_poses[i])
            rays_o, rays_d = get_rays(self.img_wh[0], self.img_wh[1], c2w)  # both (h*w, 3)
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
            rays_all.append(rays)
        return rays_all, torch.FloatTensor(val_times)

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            sample = {
                "rays": self.all_rays[idx],
                "imgs": self.all_imgs[idx],
                "time": self.all_times[idx],
            }
        else:  # create data for each image separately
            img = self.all_imgs[idx]
            rays = self.all_rays[idx]
            time = self.all_times[idx]
            sample = {"rays": rays, "imgs": img, "time": time}
        return sample

    def get_random_pose(self, batch_size, patch_size, batching="all_images"):
        """
        Apply Geometry Regularization from RegNeRF.
        This function randomly samples many patches from random poses.
        """
        n_patches = batch_size // (patch_size**2)

        N_random = self.random_rays.shape[0]
        # Sample images
        if batching == "all_images":
            idx_img = np.random.randint(0, N_random, size=(n_patches, 1))
        elif batching == "single_image":
            idx_img = np.random.randint(0, N_random)
            idx_img = np.full((n_patches, 1), idx_img, dtype=np.int)
        else:
            raise ValueError("Not supported batching type!")
        idx_img = torch.Tensor(idx_img).long()
        H, W = self.random_rays[0].shape[0], self.random_rays[0].shape[1]
        # Sample start locations
        x0 = np.random.randint(
            int(W // 4), int(W // 4 * 3) - patch_size + 1, size=(n_patches, 1, 1)
        )
        y0 = np.random.randint(
            int(H // 4), int(H // 4 * 3) - patch_size + 1, size=(n_patches, 1, 1)
        )
        xy0 = np.concatenate([x0, y0], axis=-1)
        patch_idx = xy0 + np.stack(
            np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing="xy"),
            axis=-1,
        ).reshape(1, -1, 2)

        patch_idx = torch.Tensor(patch_idx).long()
        # Subsample images
        out = self.random_rays[idx_img, patch_idx[..., 1], patch_idx[..., 0]]

        return out, self.random_times[idx_img]
