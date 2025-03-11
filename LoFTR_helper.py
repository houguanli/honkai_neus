"""
this file implement the interface for LoFTR and eLoFTR, which can detect key points pair for reg

"""

import torch
import cv2
import numpy as np
import matplotlib.cm as cm


from similar_cam_pose_detect import find_near_camera_poses, find_K_near_camera_poses
from dependencies.eloftr.src.utils.plotting import make_matching_figure
def run_loftr_inference(
    img0_path: str,
    img1_path: str,
    mask0_path: str = None,
    mask1_path: str = None,
    out_size_img=(1024, 1024),
    out_size_mask=(128, 128),
    use_eloftr = True,
    write_out_flag=False
):
    # 1) Depending on 'use_eloftr', import the corresponding LoFTR and default config.
    if use_eloftr:
        from dependencies.eloftr.src.loftr import LoFTR, full_default_cfg
        print("using eloftr")
        checkpoint_path: str = "./dependencies/eloftr/weights/eloftr_outdoor.ckpt"
        matcher = LoFTR(config=full_default_cfg)
    else:
        from dependencies.LoFTR.src.loftr import LoFTR, default_cfg
        checkpoint_path: str = "./dependencies/LoFTR/weights/indoor_ds_new.ckpt"
        matcher = LoFTR(config=default_cfg)
    state_dict = torch.load(checkpoint_path, weights_only=False)
    matcher.load_state_dict(state_dict['state_dict'], strict=True)
    matcher = matcher.eval().cuda()

    img0_raw = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    if img0_raw is None or img1_raw is None:
        print(img0_path, " ", img1_path)
        raise FileNotFoundError("无法读取图像，请检查路径是否正确。")

    h0_orig, w0_orig = img0_raw.shape[:2]
    h1_orig, w1_orig = img1_raw.shape[:2]

    # 3) 按指定大小缩放图像
    #    注意: cv2.resize 的参数顺序为 (width, height)
    img0_resized = cv2.resize(img0_raw, out_size_img)
    img1_resized = cv2.resize(img1_raw, out_size_img)

    # 4) 若有 mask，则读取并缩放
    use_mask = (mask0_path is not None) and (mask1_path is not None)
    if use_mask:
        mask0_raw = cv2.imread(mask0_path, cv2.IMREAD_GRAYSCALE)
        mask1_raw = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
        if mask0_raw is None or mask1_raw is None:
            raise FileNotFoundError("无法读取 mask，请检查路径是否正确。")

        mask0_resized = cv2.resize(mask0_raw, out_size_mask)
        mask1_resized = cv2.resize(mask1_raw, out_size_mask)
    else:
        mask0_resized, mask1_resized = None, None

    # 5) 转为 PyTorch 张量并归一化到 [0,1]
    #    shape: (1, 1, H, W)
    img0_t = torch.from_numpy(img0_resized)[None][None].float().cuda() / 255.
    img1_t = torch.from_numpy(img1_resized)[None][None].float().cuda() / 255.

    # 6) 组装 batch
    if use_mask:
        mask0_t = torch.from_numpy(mask0_resized)[None].cuda()
        mask1_t = torch.from_numpy(mask1_resized)[None].cuda()
        batch = {
            'image0': img0_t,
            'image1': img1_t,
            'mask0': mask0_t,
            'mask1': mask1_t
        }
    else:
        batch = {
            'image0': img0_t,
            'image1': img1_t
        }

    # 7) 推理，得到匹配结果
    with torch.no_grad():
        matcher(batch)

    # mkpts0_f, mkpts1_f, mconf 分别是匹配后的坐标与置信度
    mkpts0 = batch['mkpts0_f'].cpu().numpy()  # shape: (N, 2)
    mkpts1 = batch['mkpts1_f'].cpu().numpy()  # shape: (N, 2)
    mconf = batch['mconf'].cpu().numpy()      # shape: (N,)

    # 8) 将匹配坐标从 (out_size_img) 映射回原图分辨率
    #    out_size_img = (width_resize, height_resize)
    w_resize, h_resize = out_size_img
    scale_x0 = w0_orig / w_resize
    scale_y0 = h0_orig / h_resize
    mkpts0[:, 0] *= scale_x0
    mkpts0[:, 1] *= scale_y0

    scale_x1 = w1_orig / w_resize
    scale_y1 = h1_orig / h_resize
    mkpts1[:, 0] *= scale_x1
    mkpts1[:, 1] *= scale_y1

    # 9) 拼装输出坐标 => (N, 2, 2)
    #    matches[i, 0] = [x0, y0], matches[i, 1] = [x1, y1]
    matches = np.stack([mkpts0, mkpts1], axis=1).astype(np.uint)


    if write_out_flag:
        color = cm.jet(mconf, alpha=0.7)
        text = [
            'LoFTR',
            'Matches: {}'.format(len(mkpts0)),
        ]
        make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, mkpts0, mkpts1, text, path="./exp/LoFTR/e-LoFTR-keli_pose12.pdf")
    return matches, mconf


import os

def filter_and_collect_pose_pairs(
    npz1_path: str,
    images1_folder: str,
    masks1_folder: str,
    npz2_path: str,
    images2_folder: str,
    masks2_folder: str,
    delta_RT: torch.tensor,
    K: int = 0,
    file_ext: str = ".png"
):
    """
    Filters out all possible corresponding pose/image pairs between two .npz files
    (each containing multiple camera poses) and their corresponding image/mask folders.
    If K == 0, it only uses the threshold-based method (find_near_camera_poses).
    Otherwise, it uses both the threshold-based method and the top-K method
    (find_K_near_camera_poses) and merges the results.

    Args:
        npz1_path (str): Path to the first .npz file containing multiple camera poses.
        images1_folder (str): Path to the folder containing the images associated with npz1.
        masks1_folder (str): Path to the folder containing the masks associated with npz1.
        npz2_path (str): Path to the second .npz file containing multiple camera poses.
        images2_folder (str): Path to the folder containing the images associated with npz2.
        masks2_folder (str): Path to the folder containing the masks associated with npz2.
        delta_RT: Parameter(s) used by the extrinsic comparison logic
                  (e.g., rotation/translation thresholds). Actual type depends on your code.
        K (int, optional): If 0, only the threshold-based method is used. If > 0,
                           the threshold-based pairs are merged with the top-K pairs.
                           Defaults to 0.
        file_ext (str, optional): File extension for both images and masks. Defaults to ".png".

    Returns:
        list of dict: A list of dictionaries. Each dictionary describes a matching pair
                      of camera poses and their corresponding image/mask file paths. Example:
            {
                "idx1": i,  # index of pose i in npz1
                "idx2": j,  # index of pose j in npz2
                "img1": "<images1_folder>/<i>.png",
                "mask1": "<masks1_folder>/<i>.png",
                "img2": "<images2_folder>/<j>.png",
                "mask2": "<masks2_folder>/<j>.png"
            }
    """

    # 1) If K == 0, only use the threshold-based approach.
    #    Otherwise, use both threshold-based and top-K methods, then merge.
    if K == 0:
        # Only threshold-based pairs
        combined_pairs = find_near_camera_poses(npz1_path, npz2_path, delta_RT)
    else:
        # Top-K pairs
        top_k_pairs = find_K_near_camera_poses(npz1_path, npz2_path, delta_RT, K=K)
        # Merge results (remove duplicates using a set)
        combined_pairs = top_k_pairs
    results = []
    for (i, j) in combined_pairs: # assume pics are ranged as 000 version
        # Construct image and mask file names by index
        img1_path = os.path.join(images1_folder, f"{i:03}{file_ext}")
        mask1_path = os.path.join(masks1_folder, f"{i:03}{file_ext}")

        img2_path = os.path.join(images2_folder, f"{j:03}{file_ext}")
        mask2_path = os.path.join(masks2_folder, f"{j:03}{file_ext}")

        results.append({
            "idx1": i,
            "idx2": j,
            "img1": img1_path,
            "mask1": mask1_path,
            "img2": img2_path,
            "mask2": mask2_path
        })

    return results


# -------------------- 以下是一个简单的使用示例 --------------------
if __name__ == "__main__":
    # 示例：仅供参考，需要自行修改路径
    img0_path  = "./public_data/keli_iphone_pos2/image/001.png"
    img1_path  = "./public_data/keli_iphone_pos1/image/000.png"
    mask0_path = "./public_data/keli_iphone_pos2/mask/027.png"
    mask1_path = "./public_data/keli_iphone_pos1/mask/021.png"


    img0_dir_path  = "./public_data/keli_iphone_pos1/image"
    img1_dir_path  = "./public_data/keli_iphone_pos2/image"
    mask0_dir_path = "./public_data/keli_iphone_pos1/mask"
    mask1_dir_path = "./public_data/keli_iphone_pos2/mask"
    run_loftr_inference(img0_path=img0_path, img1_path=img1_path, write_out_flag=True)

    npz_path1, npz_path2 = "//wsl.localhost/Ubuntu-22.04/home/aoki/honkai_neus/public_data/keli_iphone_pos1/cameras_sphere.npz",\
        "//wsl.localhost/Ubuntu-22.04/home/aoki/honkai_neus/public_data/keli_iphone_pos2/cameras_sphere.npz" # for windows to visit wsl
    npz_path1, npz_path2 = "/home/aoki/honkai_neus/public_data/keli_iphone_pos2/cameras_sphere.npz",\
        "/home/aoki/honkai_neus/public_data/keli_iphone_pos1/cameras_sphere.npz"
    # Suppose delta_RT is a threshold or transform parameter required by your comparison logic.
    delta_RT = [[0.2656361080376526, -0.45560444077280526, 0.8496246533939018, -0.10947776249360613],
 [0.12074841154703944, 0.8900695978141784, 0.4395405921593288, -0.06454726871495192],
 [-0.9564817192270196, -0.014167024917251636, 0.2914481363630905, 0.09227380668230306],
 [0.0, 0.0, 0.0, 1.0]]  # Placeholder for demonstration

    # If K == 0, only the threshold-based approach is used. If K > 0, both approaches are merged.
    K = 0  # or 5, for example

    all_pairs_info = filter_and_collect_pose_pairs(
        npz1_path=npz_path1,
        images1_folder=img0_dir_path,
        masks1_folder=mask0_dir_path,
        npz2_path=npz_path2,
        images2_folder=img1_dir_path,
        masks2_folder=mask1_dir_path,
        delta_RT=delta_RT,
        K=K,
        file_ext=".png"
    )

    print(f"Found {len(all_pairs_info)} possible corresponding image pairs.")
    for info in all_pairs_info[:5]:  # Show the first 5
        print(info)