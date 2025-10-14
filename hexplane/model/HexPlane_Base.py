import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn
from torch.nn import functional as F

from hexplane.model.mlp import General_MLP


def DensityRender(
    xyz_sampled: torch.Tensor,
    viewdirs: torch.Tensor,
    features: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    density = features
    return density

class EmptyGridMask(torch.nn.Module):
    def __init__(
        self, device: torch.device, aabb: torch.Tensor, empty_volume: torch.Tensor
    ):
        super().__init__()
        self.device = device

        self.aabb = aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0 / self.aabbSize * 2
        self.empty_volume = empty_volume.view(1, 1, *empty_volume.shape[-3:])
        self.gridSize = torch.LongTensor(
            [empty_volume.shape[-1], empty_volume.shape[-2], empty_volume.shape[-3]]
        ).to(self.device)

    def sample_empty(self, xyz_sampled):
        empty_vals = F.grid_sample(
            self.empty_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True
        ).view(-1)
        return empty_vals


class HexPlane_Base(torch.nn.Module):
    """
    HexPlane Base Class.
    """

    def __init__(
        self,
        aabb: torch.Tensor,
        gridSize: List[int],
        device: torch.device,
        time_grid: int,
        near_far: List[float],
        density_n_comp: Union[int, List[int]] = 8,
 
        density_dim: int = 1,
   
        DensityMode: str = "general_MLP",

        emptyMask: Optional[EmptyGridMask] = None,
        fusion_one: str = "multiply",
        fusion_two: str = "concat",
        fea2denseAct: str = "softplus",
        init_scale: float = 0.1,
        init_shift: float = 0.0,
        normalize_type: str = "normal",
        **kwargs,
    ):
        super().__init__()

        self.aabb = aabb
        self.device = device
        self.time_grid = time_grid
        self.near_far = near_far
        self.near_far_org = near_far

        self.step_ratio = kwargs.get("step_ratio", 2.0)
        self.update_stepSize(gridSize)
        #####################################################
        self.datasampler_type = kwargs.get(
            "datasampler_type",
        ) 
        #####################################################
        self.density_n_comp = density_n_comp
 
        self.density_dim = density_dim

        self.align_corners = kwargs.get(
            "align_corners", True
        )  # align_corners for grid_sample

        # HexPlane weights initialization: scale and shift for uniform distribution.
        self.init_scale = init_scale
        self.init_shift = init_shift

        # HexPlane fusion mode.
        self.fusion_one = fusion_one
        self.fusion_two = fusion_two

        # Plane Index
        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]

        # Coordinate normalization type.
        self.normalize_type = normalize_type

        # Plane initialization.
        self.init_planes(gridSize[0], device)

        # Density calculation settings.
        self.fea2denseAct = fea2denseAct  # feature to density activation function
        self.density_shift = kwargs.get(
            "density_shift", 0
        )  # density shift for density activation function.
        self.distance_scale = kwargs.get(
            "distance_scale", 25.0
        )  # distance scale for density activation function.
        self.DensityMode = DensityMode
        self.density_t_pe = kwargs.get("density_t_pe", -1)
        self.density_pos_pe = kwargs.get("density_pos_pe", -1)
        self.density_view_pe = kwargs.get("density_view_pe", -1)
        self.density_fea_pe = kwargs.get("density_fea_pe", 2)
        self.density_featureC = kwargs.get("density_featureC", 128)
        self.density_n_layers = kwargs.get("density_n_layers", 3)
        self.init_density_func(
            self.DensityMode,
            self.density_t_pe,
            self.density_pos_pe,
            self.density_view_pe,
            self.density_fea_pe,
            self.density_featureC,
            self.density_n_layers,
            self.device,
        )


    def init_density_func(
        self, DensityMode, t_pe, pos_pe, view_pe, fea_pe, featureC, n_layers, device
    ):
        """
        Initialize density regression function.
        """
        if (
            DensityMode == "plain"
        ):  # Use extracted features directly from HexPlane as density.
            assert self.density_dim == 1  # Assert the extracted features are scalers.
            self.density_regressor = DensityRender
        elif DensityMode == "general_MLP":  # Use general MLP to regress density.
            assert (
                view_pe < 0
            )  # Assert no view position encoding. Density should not depend on view direction.
            self.density_regressor = General_MLP(
                self.density_dim,
                1,
                t_pe,
                fea_pe,
                pos_pe,
                view_pe,
                featureC,
                n_layers,
                use_sigmoid=False,
                zero_init=False,
            ).to(device)
        else:
            raise NotImplementedError("No such Density Regression Mode")
        print("DENSITY REGRESSOR:")
        print(self.density_regressor)
        print("t_pe", t_pe, "pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0 / self.aabbSize
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        self.units = self.aabbSize / (self.gridSize - 1)
        self.stepSize = torch.mean(self.units) * self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_planes(self, res, device):
        pass

    def compute_features(self, xyz_sampled):
        pass

    def compute_densityfeature(self, xyz_sampled, frame_time):
        pass


    def normalize_coord(self, xyz_sampled):
        """
        Normalize the sampled coordinates to [-1, 1] range.
        """
        if self.normalize_type == "normal":
            return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1

    def feature2density(self, density_features: torch.Tensor) -> torch.Tensor:
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features+self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)
        else:
            raise NotImplementedError("No such activation function for density feature")

    def sample_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        is_train: bool = True,
        N_samples: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample points along rays based on the given ray origin and direction.

        Args:
            rays_o: (B, 3) tensor, ray origin.
            rays_d: (B, 3) tensor, ray direction.
            is_train: bool, whether in training mode.
            N_samples: int, number of samples along each ray.

        Returns:
            rays_pts: (B, N_samples, 3) tensor, sampled points along each ray.
            ~mask_outbbox: (B, N_samples) tensor, mask for points within bounding box.
            z_vals: (B, N_samples) tensor, sampled points' distance to each ray origin
        """

        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        z_near = near * torch.ones_like(rays_d[..., :1]) # b,1
        z_far = far * torch.ones_like(rays_d[..., :1])   # b,1
        
        interpx = torch.linspace(0, 1, N_samples).unsqueeze(0).to(rays_o) #N_samples
        z_vals = z_near * (1.0 - interpx) + z_far * interpx  # b, N_samples
            # Get intervals between samples.
        if is_train:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
            lower = torch.cat((z_vals[..., :1], mids), dim=-1)
            # Stratified samples in those intervals.
            t_rand = torch.rand(z_vals.shape, dtype=torch.float, device=self.device)
            z_vals = lower + (upper - lower) * t_rand # b, N_samples

        


        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]
 
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(
            dim=-1
        )
        return rays_pts, ~mask_outbbox, z_vals

    def forward(
        self,
        rays_chunk: torch.Tensor,
        frame_time: torch.Tensor,
        white_bg: bool = True,
        is_train: bool = False,
        ndc_ray: bool = False,
        N_samples: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the HexPlane.

        Args:
            rays_chunk: (B, 6) tensor, rays [rays_o, rays_d].
            frame_time: (B, 1) tensor, time values.
            white_bg: bool, whether to use white background.
            is_train: bool, whether in training mode.
            ndc_ray: bool, whether to use normalized device coordinates.
            N_samples: int, number of samples along each ray.

        Returns:
            rgb: (B, 3) tensor, rgb values.
            depth: (B, 1) tensor, depth values.
            alpha: (B, 1) tensor, accumulated weights.
            z_vals: (B, N_samples) tensor, z values.
        """
        # Prepare rays.


        viewdirs = rays_chunk[:, 3:6]
        xyz_sampled, ray_valid, z_vals_s = self.sample_rays(
            rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples
        )
        #s_ray_valid = torch.sum(ray_valid)
        #num_ray_valid = torch.tensor(ray_valid.shape).prod()
        #print(f's_ray_valid:{s_ray_valid}, {num_ray_valid}')
        #np.save('/gpfs/offline1/staff/tomograms/users/zhehu/Hexplane/ray_o/xyz_sampled.npy',xyz_sampled.cpu().numpy())
        dists = torch.cat(
            (z_vals_s[:, 1:] - z_vals_s[:, :-1], torch.zeros_like(z_vals_s[:, :1])), dim=-1
        )
        rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
        if ndc_ray:
            dists = dists * rays_norm

        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        frame_time = frame_time.view(-1, 1, 1).expand(
            xyz_sampled.shape[0], xyz_sampled.shape[1], 1
        )

        # Normalize coordinates.
        #xyz_sampled = self.normalize_coord(xyz_sampled)

        # Initialize sigma and rgb values.
        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        atten_map = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        # Compute density feature and density if there are valid rays.
        if ray_valid.any():
            
            density_feature = self.compute_densityfeature(
                xyz_sampled[ray_valid], frame_time[ray_valid]
            )
            #print(f'density_feature1:{density_feature.shape}')
            density_feature = self.density_regressor(
                xyz_sampled[ray_valid],
                viewdirs[ray_valid],
                density_feature,
                frame_time[ray_valid],
            )
            #print(f'density_feature2:{density_feature.shape}')
            validsigma = self.feature2density(density_feature)
            #print(f'validsigma:{validsigma.shape}')
            sigma[ray_valid] = validsigma.view(-1)


        dists = dists * rays_norm
        #print(f'z_vals_s shape: {z_vals_s.shape}, dists shape:{dists.shape},viewdirs shape:{viewdirs.shape}, sigma shape:{sigma.shape},dists_s shape:{dists_s.shape},')
        atten_map = (sigma*dists).sum(-1)



        return atten_map, sigma*dists

