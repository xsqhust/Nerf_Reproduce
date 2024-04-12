from typing_extensions import Self
import torch.utils.data as data
import numpy as np
import os
from lib.utils import data_utils
from lib.config import cfg
from torchvision import transforms as T
import imageio
import json
import cv2
import torch

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

#=============获取相机姿态矩阵===================
def pose_spherical(theta, phi, radius):
    #theta为仰角；phi为方位角；radius为距离球心距离
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w
#=================================================

#=============根据图像长度宽度、焦距以及相机姿态确定射线===================
def get_rays(H, W, K, c2w):
    #图像长度和宽度；焦距；姿态矩阵
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d
#==========================================================================


#=============根据图像长度宽度、焦距以及相机姿态确定射线_numpy===================
def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d
#===============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()

        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene
        self.input_ratio = kwargs['input_ratio']
        self.data_root = os.path.join(data_root, scene)
        self.batch_size = cfg.task_arg.N_rays
        self.split = split
        self.K = None
        self.iter_num = 0
        self.precrop_iters = cfg.task_arg.precrop_iters
        self.precrop_frac = cfg.task_arg.precrop_frac
        #========读取图像以及相机姿态信息============
        skip = 1
        imgs = []
        poses = []
        image_paths = []
        json_info = json.load(open(os.path.join(self.data_root, 'transforms_{}.json'.format('train'))))
        for frame in json_info['frames'][::skip]:
            fname = os.path.join(self.data_root, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # (100,800,800,4)
        self.num_samples = imgs.shape[0]
        poses = np.array(poses).astype(np.float32)# (100,4,4)

        H, W = imgs[0].shape[:2]
        camera_angle_x = float(json_info['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

        #========缩小图像尺寸============
        if self.input_ratio !=1 :
            H = H//2
            W = W//2
            focal = focal/2.

            imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
            for i, img in enumerate(imgs):
                imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            imgs = imgs_half_res

        self.H = H
        self.W = W
        self.poses =poses


         #========是否白色背景============
        if cfg.task_arg.white_bkgd:
            imgs= imgs[...,:3]*imgs[...,-1:] + (1.-imgs[...,-1:])
        else:
            imgs = imgs[...,:3]
        self.imgs = imgs
        #======近平面以及远平面=============
        self.near = 2.
        self.far = 6.


        #======相机内参矩阵===========
        if self.K is None:
            self.K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])

        #======定义所有的光线起点以及方向==========
        rays_o, rays_d = [], []
        for i in range(self.num_samples):
            pose = self.poses[i, :3,:4]
            ray_o, ray_d = get_rays_np(self.H, self.W, self.K, pose)
            rays_d.append(ray_d)   # (400, 400, 3)
            rays_o.append(ray_o)   # (400, 400, 3)
        self.rays_o= np.array(rays_o) # (100, 400, 400, 3)
        self.rays_d= np.array(rays_d) # (100, 400, 400, 3)



    def __getitem__(self, index):

        if self.split == 'train':
            H=self.H
            W=self.W
            self.iter_num += 1
            if self.iter_num< self.precrop_iters:
                dH = int(H//2 * cfg.task_arg.precrop_frac)
                dW = int(W//2 * cfg.task_arg.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H//2 - dH, self.H//2 + dH - 1, 2*dH), 
                        torch.linspace(W//2 - dW, self.W//2 + dW - 1, 2*dW)
                    ), -1)
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
            coords = torch.reshape(coords, [-1, 2])

            #==========选取一个batchsize的射线进行训练=================
            select_ids = np.random.choice(coords.shape[0], size=self.batch_size, replace=False)
            select_coords = coords[select_ids].long()


            #=======某一个位姿所对应的射线原点、方向以及图像rgb真值==============self.W
            ray_os = self.rays_o[index]  # (H, W, 3)
            ray_ds = self.rays_d[index]  # (H, W, 3)
            rgbs = self.imgs[index]      # (H, W, 3)

            #=======所挑选的一个bactsize射线所对应的射线原点方向以及像素rgb真值
            ray_o = ray_os[select_coords[:, 0], select_coords[:, 1]]  # (N_rays, 3)
            ray_d = ray_ds[select_coords[:, 0], select_coords[:, 1]]  # (N_rays, 3)
            rgb = rgbs[select_coords[:, 0], select_coords[:, 1]]      # (N_rays, 3)

        else:
            #======================测试阶段对所有的射线所对应的像素值进行预测==================
            ray_o = self.rays_o[index].reshape(-1, 3)  # (H * W, 3)
            ray_d = self.rays_d[index].reshape(-1, 3)  # (H * W, 3)
            rgb = self.imgs[index].reshape(-1, 3)      # (H * W, 3)
        ret = {'ray_o': ray_o, 'ray_d': ray_d, 'rgb': rgb, 'near':self.near, 'far':self.far,  'H': self.H,
                'W': self.W,}
        return ret



    def __len__(self):
        # 返回训练样本的个数
        return self.num_samples
