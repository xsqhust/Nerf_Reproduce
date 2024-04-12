import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.encoding import get_encoder
from lib.config import cfg

from .run_nerf_helpers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, skips=[4], use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.output_ch = 5 if self.use_viewdirs else 4
        # 定义位置编码后的位置信息的线性层，如果层数在skips列表中，则将原始位置信息与隐藏层拼接
        self.pts_linears = nn.ModuleList(
        [nn.Linear(self.input_ch, self.W)] + [nn.Linear(self.W, self.W) if i not in self.skips else nn.Linear(self.W + self.input_ch, self.W) for i in
                                        range(self.D - 1)])
        # 定义位置编码后的视角方向信息的线性层
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_views + self.W, self.W // 2)])

        if self.use_viewdirs:
            # feature vector(256)
             # 定义特征向量的线性层
            self.feature_linear = nn.Linear(self.W, self.W)
             # 定义透明度（alpha）值的线性层
            self.alpha_linear = nn.Linear(self.W, 1)
            # 定义RGB颜色的线性层
            self.rgb_linear = nn.Linear(self.W // 2, 3)
        else:
            # output channel(default: 4)
            self.output_linear = nn.Linear(self.W, self.output_ch)



    def forward(self, x):
        # 提取位置和视角方向信息
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        # ======位置信息输入网络========
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)  # 如果层数在skips列表中，则将原始位置信息与隐藏层拼接
        # ======使用方向信息========
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)  # 预测透明度
            feature = self.feature_linear(h)  # 预测特征向量
            h = torch.cat([feature, input_views], -1)# 将特征向量与视觉方向拼接
             # ======方向信息输入网络========
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

        #=========预测rgb并将rgb和透明度进行拼接
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.N_samples = cfg.task_arg.N_samples # 粗采样点
        self.N_importance = cfg.task_arg.N_importance # 精细采样点
        self.chunk = cfg.task_arg.chunk_size # 并行处理4096个射线  
        self.batch_size = cfg.task_arg.N_rays # batchsize训练大小为  1024个射线  
        self.white_bkgd = cfg.task_arg.white_bkgd # 用白色背景
        self.use_viewdirs = cfg.task_arg.use_viewdirs # 是否加入相机方向信息
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # =========================对输入相机位置以及方向进行编码=========================
        self.embed_fn, self.input_ch = get_encoder(cfg.network.xyz_encoder)
        self.embeddirs_fn, self.input_ch_views = get_encoder(cfg.network.dir_encoder)

        # coarse model
        self.model = NeRF(D=cfg.network.nerf.D_c,
                          W=cfg.network.nerf.W_c,
                          input_ch=self.input_ch,
                          input_ch_views=self.input_ch_views,
                          skips=cfg.network.nerf.skips,
                          use_viewdirs=self.use_viewdirs)

        # fine model
        self.model_fine = NeRF(D=cfg.network.nerf.D_f,
                               W=cfg.network.nerf.W_f,
                               input_ch=self.input_ch,
                               input_ch_views=self.input_ch_views,
                               skips=cfg.network.nerf.skips,
                               use_viewdirs=self.use_viewdirs)


    def batchify(self, fn, chunk):
        """Constructs a version of 'fn' that applies to smaller batches."""
        def ret(inputs):
            return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
        return ret


    def forward_network(self, inputs, viewdirs, model=''):
        """Prepares inputs and applies network 'fn'."""
        if model == 'fine':
            fn = self.model_fine
        else:
            fn = self.model

        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = self.embed_fn(inputs_flat)

        if self.use_viewdirs:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        embedded = embedded.to(torch.float32)
        outputs_flat = self.batchify(fn, self.chunk)(embedded)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs


    def render_rays(self,ray_batch,
        N_samples= cfg.task_arg.N_samples,  #每条光线的采样数
        retraw=False,
        lindisp=cfg.task_arg.lindisp,
        perturb=cfg.task_arg.perturb,
        N_importance=cfg.task_arg.N_importance,
        white_bkgd= cfg.task_arg.white_bkgd,
        raw_noise_std= cfg.task_arg.raw_noise_std,
        pytest=False):
        """Volumetric rendering.
        Args:
        ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
        network_fn: function. Model for predicting RGB and density at each point
            in space.
        network_query_fn: function used for passing queries to network_fn.
        N_samples: int. Number of different times to sample along each ray.
        retraw: bool. If True, include model's raw, unprocessed predictions.
        lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
        perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
        N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
        network_fine: "fine" network with same spec as network_fn.
        white_bkgd: bool. If True, assume a white background.
        raw_noise_std: ...
        verbose: bool. If True, print more debugging info.
        Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
        disp_map: [num_rays]. Disparity map. 1 / depth.
        acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
        raw: [num_rays, num_samples, 4]. Raw predictions from model.
        rgb0: See rgb_map. Output for coarse model.
        disp0: See disp_map. Output for coarse model.
        acc0: See acc_map. Output for coarse model.
        z_std: [num_rays]. Standard deviation of distances along ray for each
            sample.
        """
        
        N_rays = ray_batch.shape[0]   #射线得数量
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] 射线得起点以及方向
        viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None   # 方向信息
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])   
        near, far = bounds[...,0], bounds[...,1] # [-1,1]

        t_vals = torch.linspace(0., 1., steps=N_samples)
        t_vals = t_vals.to(device)
        if not lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)
            t_rand = t_rand.to(device)
            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


    #     raw = run_network(pts)
        raw = self.forward_network(pts, viewdirs)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

        if N_importance > 0:

            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

           
    #         raw = run_network(pts, fn=run_fn)
            raw = self.forward_network(pts, viewdirs, model='fine')

            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

        ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
        if retraw:
            ret['raw'] = raw
        if N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        for k in ret:
            DEBUG= False
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
                print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret



    def batchify_rays(self,rays_flat, chunk=1024*32):
        """Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = {}
        for i in range(0, rays_flat.shape[0], chunk):
            ret = self.render_rays(rays_flat[i:i+chunk])
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret


    def render(self,batch):
        """Render rays
        Args:
        H: int. Height of image in pixels.
        W: int. Width of image in pixels.
        focal: float. Focal length of pinhole camera.
        chunk: int. Maximum number of rays to process simultaneously. Used to
            control maximum memory usage. Does not affect final results.
        rays: array of shape [2, batch_size, 3]. Ray origin and direction for
            each example in batch.
        c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
        ndc: bool. If True, represent ray origin, direction in NDC coordinates.
        near: float or array of shape [batch_size]. Nearest distance for a ray.
        far: float or array of shape [batch_size]. Farthest distance for a ray.
        use_viewdirs: bool. If True, use viewing direction of a point in space in model.
        c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
        camera while using other c2w argument for viewing directions.
        Returns:
        rgb_map: [batch_size, 3]. Predicted RGB values for rays.
        disp_map: [batch_size]. Disparity map. Inverse of depth.
        acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
        extras: dict with everything returned by render_rays().
        """
        rays_o = batch['ray_o']
        rays_d = batch['ray_d']
        near = batch['near']
        far = batch['far']

        viewdirs = rays_d 
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

        sh = rays_d.shape # [..., 3]
        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()

        near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        if self.use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)

        # Render and reshape
        all_ret = self.batchify_rays(rays,self.chunk)
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        # k_extract = ['rgb_map', 'disp_map', 'acc_map']
        # ret_list = [all_ret[k] for k in k_extract]
        # ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
        return all_ret
    #===============================体渲染======================














