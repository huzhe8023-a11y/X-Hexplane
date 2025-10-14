import os

import imageio
import numpy as np
import torch
from pytorch_msssim import ms_ssim as MS_SSIM
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from hexplane.render.util.metric import rgb_lpips, rgb_ssim
from hexplane.render.util.util import visualize_depth_numpy

def save_tensor_plot(data, savePath=0, prefix=0, idx=0, gt=False):
    plt.rcParams.update({"font.size": 22})
    plt.figure(figsize=(20, 20))
    plt.imshow(data)
    plt.axis("off")
    if gt:
        plt.savefig(f"{savePath}/{prefix}{idx:03d}_gt.png")
    else:
        plt.savefig(f"{savePath}/{prefix}{idx:03d}.png")
    plt.cla()
    plt.close()
        
        
def OctreeRender_trilinear_fast(
    rays,
    time,
    model,
    chunk=4096,
    N_samples=-1,
    ndc_ray=False,
    white_bg=False,
    is_train=False,
    device="cuda",
):
    """
    Batched rendering function.
    """
    atten_maps, z_vals = [], []
    N_rays_all = rays.shape[0]
    if model.datasampler_type == 'images':
        rays = rays.reshape(-1, rays.shape[-1])
        print(f'rays.shape:{rays.shape}')
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        time_chunk = time[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)

        atten_map, z_val_map = model(
            rays_chunk,
            time_chunk,
            is_train=is_train,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            N_samples=N_samples,
        )
        atten_maps.append(atten_map)
        z_vals.append(z_val_map)
    return (
        torch.cat(atten_maps),
        torch.cat(z_vals),
    )


@torch.no_grad()
def evaluation(
    test_dataset,
    model,
    cfg,
    savePath=None,
    N_vis=5,
    prefix="",
    N_samples=-1,
    white_bg=False,
    ndc_ray=False,
    compute_extra_metrics=False,
    device="cuda",
):
    """
    Evaluate the model on the test rays and compute metrics.
    """
    PSNRs, atten_maps, LOSS, thD_list = [], [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/atten", exist_ok=True)
    try:
        tqdm._instances.clear()
    except Exception:
        pass

    w,h = test_dataset.img_wh
    lenth = len(test_dataset.theta_y)
    #img_eval_interval = lenth if N_vis < 0 else max(len(test_dataset) // lenth, 1)
    img_eval_interval = lenth
    #idxs = list(range(0, len(test_dataset)//50, img_eval_interval))
    idxs = list(range(2000, 2800, img_eval_interval))
    #print(f'idxs:{idxs}')0
    for idx in tqdm(idxs):
        idx = int(idx)
        data = test_dataset[idx]
        samples, gt_img, sample_times = data["rays"], data["imgs"], data["time"]

        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])
        times = sample_times.view(-1, sample_times.shape[-1])
        atten_map, z_vals = OctreeRender_trilinear_fast(
            rays,
            times,
            model,
            chunk=4096,
            N_samples=N_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )
        atten_map = atten_map.reshape(H, W).cpu()
        z_vals = z_vals.reshape(H,W,N_samples).cpu().numpy()
        z_vals = z_vals[...,N_samples//4:N_samples//4+N_samples//2]

        thD_list.append(z_vals)
        if len(test_dataset):
            gt_img = gt_img.view(H, W)

            loss = torch.mean((atten_map - gt_img) ** 2)
            LOSS.append(loss)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))




        atten_map = atten_map.numpy()

        #gt_img_map = gt_img.numpy().astype(np.float32)
        gt_img_map = gt_img.numpy()
        #gt_img_map = (gt_img_map * 255)


        if savePath is not None:
            save_tensor_plot(atten_map,savePath,prefix,idx)
            save_tensor_plot(gt_img_map,savePath,prefix,idx,gt=True)
            imageio.imwrite(f"{savePath}/atten/{prefix}{idx:03d}.png", atten_map)



    
    fourda = np.stack(thD_list)

    np.save(f"{savePath}/atten/{prefix}4d_att.npy", fourda)
    
    
    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            pass
        else:
            with open(f"{savePath}/{prefix}mean.txt", "w") as f:
                f.write(f"PSNR: {psnr} \n")
                print(f"PSNR: {psnr} \n")
                for i in range(len(PSNRs)):
                    f.write(f"Index {i}, PSNR: {PSNRs[i]}\n")

    return PSNRs, LOSS


@torch.no_grad()
def evaluation_path(
    test_dataset,
    model,
    cfg,
    savePath=None,
    N_vis=5,
    prefix="",
    N_samples=-1,
    white_bg=False,
    ndc_ray=False,
    compute_extra_metrics=True,
    device="cuda",
):
    """
    Evaluate the model on the valiation rays.
    """
    img_maps, atten_maps, depth_maps = [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/imgd", exist_ok=True)
    os.makedirs(savePath + "/atten", exist_ok=True)
    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    val_rays, val_times = test_dataset.get_val_rays()

    for idx in tqdm(range(val_times.shape[0])):
        W, H = test_dataset.img_wh
        #W, H = 256, 256
        rays = val_rays[idx]
        time = val_times[idx]
        time = time.expand(rays.shape[0], 1)
        img_map, atten_map, depth_map, _, _ = OctreeRender_trilinear_fast(
            rays,
            time,
            model,
            chunk=8192,
            N_samples=N_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )
        img_map = img_map.clamp(0.0, 1.0)

        img_map, atten_map, depth_map = (
            img_map.reshape(H, W, 1).cpu(),
            atten_map.reshape(H, W, 1).cpu(),
            depth_map.reshape(H, W).cpu(),
        )

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

        img_map = (img_map.numpy() * 255).astype("uint8")
        atten_map = atten_map.numpy()
        img_maps.append(img_map)
        atten_maps.append(atten_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            save_tensor_plot(img_map,savePath,prefix,idx)
            #img_map = np.concatenate((img_map, depth_map), axis=1)
            imageio.imwrite(f"{savePath}/atten/{prefix}{idx:03d}.png", atten_map)

    imageio.mimwrite(
        f"{savePath}/{prefix}video.gif", np.stack(img_maps), fps=2
    )
    imageio.mimwrite(
        f"{savePath}/{prefix}attenvideo.gif", np.stack(atten_maps), fps=2
    )

    return 0
