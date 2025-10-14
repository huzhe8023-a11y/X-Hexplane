from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class System_Config:
    seed: int = 20231009
    basedir: str = "/data/staff/tomograms/users/kallejosefsson/Hexplane/results"
    ckpt: Optional[str] = None
    progress_refresh_rate: int = 100
    vis_every: int = 5000
    add_timestamp: bool = True


@dataclass
class Model_Config:
    model_name: str = "HexPlane"  # choose from "HexPlane", "HexPlane_Slim"
    N_voxel_init: int = 64 * 64 * 64  # initial voxel number
    N_voxel_final: int = 128 * 128 * 128# final voxel number
    step_ratio: float = 1
    aspect_ratio: float = 3
    nonsquare_voxel: bool = True  # if yes, voxel numbers along each axis depend on scene length along each axis
    time_grid_init: int = 32
    time_grid_final: int = 64
    normalize_type: str = "normal"
    upsample_list: List[int] = field(default_factory=lambda: [3000, 6000, 9000])
    update_emptymask_list: List[int] = field(
        default_factory=lambda: [4000, 8000, 10000]
    )

    # Plane Initialization
    density_n_comp: List[int] = field(default_factory=lambda: [36, 36, 36])

    density_dim: int = 15

    DensityMode: str = "general_MLP"  # choose from "plain", "general_MLP"

    init_scale: float = 0.1
    init_shift: float = 0.0

    # Fusion Methods
    fusion_one: str = "multiply"
    fusion_two: str = "concat"

    # Density Feature Settings
    fea2denseAct: str = "softplus"
    density_shift: float = 0
    distance_scale: float = 1

    # Density Regressor MLP settings
    density_t_pe: int = -1
    density_pos_pe: int = 5
    density_view_pe: int = -1
    density_fea_pe: int = 2
    density_featureC: int = 64
    density_n_layers: int = 3
    



    TV_t_s_ratio: float = 1   # ratio of TV loss along temporal and spatial dimensions
    TV_weight_density: float = 1
    L1_weight_density: float = 1


    # Sampling
    align_corners: bool = True
    # There are two types of upsampling: aligned and unaligned.
    # Aligned upsampling: N_t = 2 * N_t-1 - 1. Like: |--|--| ->upsampling -> |-|-|-|-|, where | represents sampling points and - represents distance.
    # Unaligned upsampling: We use linear_interpolation to get the new sampling points without considering the above rule.
    # using "unaligned" upsampling will essentially double the grid sizes at each time, ignoring N_voxel_final.
    upsampling_type: str = "aligned"  # choose from "aligned", "unaligned".
    nSamples: int = 1000000



@dataclass
class Data_Config:
    datadir: str = "/data/staff/tomograms/users/kallejosefsson/Hexplane/am_rec_raw.npy"
    dataset_name: str = "XMPIDataset"  # choose from "dnerf", "XMPIDataset", "neural3D_NDC"
    downsample: float = 1
    cal_fine_bbox:  bool = False
    N_vis: int = -1
    idx: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    num_angles: int = 8000                                              #!!!!!!!!!!!!!!!!!!!!!!!!!!
    time_scale: float = 1
    scene_bbox_min: List[float] = field(default_factory=lambda: [-1, -1, -1])
    scene_bbox_max: List[float] = field(default_factory=lambda: [1, 1, 1])
    N_random_pose: int = 1000
    datasampler_type: str = "rays"  # choose from "rays", "images", "hierach"
    size: List[int] = field(default_factory=lambda: [528, 70])         #!!!!!!!!!!!!!!!!!!!!!!!!!!
    nta: int = 4

@dataclass
class Optim_Config:
    # Learning Rate
    lr_density_grid: float = 0.04

    lr_density_nn: float = 0.002
    
    # Optimizer, Adam deault
    beta1: float = 0.9
    beta2: float = 0.99
    lr_decay_type: str = "exp"  # choose from "exp" or "cosine" or "linear"
    lr_decay_target_ratio: float = 0.9
    lr_decay_step: int = 10000
    lr_upsample_reset: bool = True
  
    batch_size: int = 4096
    n_iters: int = 100000 


@dataclass
class Config:
    config: Optional[str] = None
    expname: str = "default"

    render_only: bool = False
    render_train: bool = True
    render_test: bool = True
    render_path: bool = False

    systems: System_Config = System_Config()
    model: Model_Config = Model_Config()
    data: Data_Config = Data_Config()
    optim: Optim_Config = Optim_Config()
