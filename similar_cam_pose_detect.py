import numpy as np
import open3d as o3d
import cv2 as cv
import torch


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def evaluate_extrinsic_difference(
    M1: np.ndarray,
    M2: np.ndarray,
    rotation_threshold_deg: float = 30.0,
    translation_threshold: float = 0.5,
    is_print_info = False
) -> bool:
    """
    比较两个相机外参矩阵的差异，并判断是否在给定阈值范围内。

    参数:
    -------
    M1 : np.ndarray
        第一个外参矩阵，形状为 (4, 4)。
    M2 : np.ndarray
        第二个外参矩阵，形状为 (4, 4)。
    rotation_threshold_deg : float
        旋转角度阈值 (单位: 度)，用于判断两个矩阵的旋转差异是否可接受。
    translation_threshold : float
        平移距离阈值，用于判断两个矩阵的平移差异是否可接受。
    bool
        如果两个外参矩阵的差异（旋转与平移）都在阈值范围内，返回 True；否则返回 False。
    """

    # 1. 计算相对变换矩阵 (M_rel = M1^-1 * M2)
    #    其含义为：从 M1 对应的坐标系，变换到 M2 对应的坐标系
    M_rel = np.linalg.inv(M1) @ M2

    # 2. 提取旋转矩阵和平移向量
    R_rel = M_rel[:3, :3]   # 相对旋转
    t_rel = M_rel[:3, 3]    # 相对平移

    # 3. 计算旋转差异（以角度为单位）
    #    旋转矩阵 R 的旋转角度可通过以下公式计算：
    #    angle = arccos((trace(R) - 1) / 2)
    #    注意数值稳定性，可能需要 min/max 限制
    trace_val = np.trace(R_rel)
    # 防止浮点误差导致 arccos 参数超出 [-1, 1]
    cos_angle = max(min((trace_val - 1.0) / 2.0, 1.0), -1.0)
    rotation_angle_deg = np.degrees(np.arccos(cos_angle))

    # 4. 计算平移差异（向量范数）
    translation_diff = np.linalg.norm(t_rel)

    # 5. 根据阈值进行判断
    rotation_ok = (rotation_angle_deg <= rotation_threshold_deg)
    translation_ok = (translation_diff <= translation_threshold)
    if rotation_ok and translation_ok and is_print_info:
        print("rotation diff: ", rotation_angle_deg)
        print("translation diff ", translation_diff)
    return rotation_ok and translation_ok

def test_similar_pose_detect():
    M1_example = [[0.7423551082611084, -0.2518492341041565, -0.6208710670471191, 0.49617502093315125],
                  [-0.6630696654319763, -0.14315412938594818, -0.7347418069839478, 0.22048130631446838],
                  [0.09616389870643616, 0.9571200609207153, -0.2732647955417633, 0.20496077835559845],
                  [0.0, 0.0, 0.0, 1.0]]

    M2_example = [[-0.9261959195137024, 0.004189362749457359, -0.37701937556266785, 0.2073606550693512],
                  [0.37704265117645264, 0.010291064158082008, -0.926138699054718, 0.5093762874603271],
                  [-2.3283061589829401e-10, -0.9999382495880127, -0.011111109517514706, 0.0061111110262572765],
                  [0.0, 0.0, 0.0, 1.0]]

    RT = np.array([
        [
            -0.5849950772807017,
            0.801868264500357,
            -0.12160610981579512,
            0.2416590098834332
        ],
        [
            0.8033941419548528,
            0.5934671410150251,
            0.04852427442093907,
            -0.02324493170884405
        ],
        [
            0.11107930603839655,
            -0.06931117458706122,
            -0.9913916223407326,
            0.023850916562664706
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ])  # a test for obb based exc finding
    M1_example = RT @ M1_example  # trans with raw RT
    result = evaluate_extrinsic_difference(
        M1_example,
        M2_example,
        rotation_threshold_deg=30.0,
        translation_threshold=0.5
    )
    print("is differ:", result)

def from_npz_get_poses(npz_path):
    data = np.load(npz_path)
    poses = []
    for key in data.files:
        if key.startswith("world_mat") and not key.startswith("world_mat_inv"):
            intrinsics, pose = load_K_Rt_from_P("none", data[key][:3, :4])
            poses.append(pose)
    return poses

def find_near_camera_poses(
    npz_path1: str,
    npz_path2: str,
    delta_RT: np.array,
    rotation_threshold_deg: float = 50.0,
    translation_threshold: float = 2
):
    """
    从两个 npz 文件中读取所有相机位姿 (外参矩阵)，并找出在旋转和平移上都足够接近的配对。

    参数:
    -------
    npz_path1 : str
        第一个 .npz 文件路径。
    npz_path2 : str
        第二个 .npz 文件路径。
    rotation_threshold_deg : float
        旋转角度阈值 (单位: 度)，用于判断两个位姿的旋转差异是否可接受。
    translation_threshold : float
        平移距离阈值，用于判断两个位姿的平移差异是否可接受。

    返回:
    -------
    near_pairs : list of (int, int)
        一个列表，每个元素是 (i, j)，表示第 i 个 pose1 与第 j 个 pose2 满足阈值要求。
    """
    poses1 = from_npz_get_poses(npz_path1)  # shape: (N1, 4, 4)
    poses2 = from_npz_get_poses(npz_path2)  # shape: (N2, 4, 4)

    near_pairs = []
    for i, M1 in enumerate(poses1):
        for j, M2 in enumerate(poses2):
            delta_M1 = delta_RT @ M1
            if evaluate_extrinsic_difference(
                delta_M1, M2,
                rotation_threshold_deg=rotation_threshold_deg,
                translation_threshold=translation_threshold
            ):
                near_pairs.append((i, j))

    return near_pairs

def measure_extrinsic_difference(M1: np.ndarray, M2: np.ndarray) -> float:
    """
    计算两个外参矩阵 M1, M2 的差异度。
    这里简单地将 '旋转角度(度)' + '平移差异' 作为一个标量距离。
    """
    # 相对变换矩阵
    M_rel = np.linalg.inv(M1) @ M2
    R_rel = M_rel[:3, :3]
    t_rel = M_rel[:3, 3]

    # 计算旋转角度
    trace_val = np.trace(R_rel)
    # 为避免浮点误差，需确保 arccos 参数在 [-1, 1] 范围内
    cos_angle = max(min((trace_val - 1.0) / 2.0, 1.0), -1.0)
    rotation_angle_deg = np.degrees(np.arccos(cos_angle))

    # 计算平移差异
    translation_diff = np.linalg.norm(t_rel)

    # 将旋转与平移差异合并为单一标量
    # 可根据需求对二者加权
    difference = rotation_angle_deg + translation_diff
    return difference

def find_K_near_camera_poses(
    npz_path1: str,
    npz_path2: str,
    delta_RT:torch.Tensor,
    K: int = 5
):
    """
    在不使用阈值的情况下，寻找两个 .npz 文件中外参矩阵差异最小的 K 个配对。

    参数:
    -------
    npz_path1 : str
        第一个 .npz 文件路径。
    npz_path2 : str
        第二个 .npz 文件路径。
    K : int
        需要返回的最小差异配对数量。

    返回:
    -------
    top_K_pairs : list of (int, int, float)
        长度最多为 K 的列表，每个元素为 (i, j, diff)，
        表示第 i 个 pose1 与第 j 个 pose2 的差异度为 diff。
        该列表按 diff 从小到大排序。
    """
    poses1 = from_npz_get_poses(npz_path1)  # shape: (N1, 4, 4)
    poses2 = from_npz_get_poses(npz_path2)  # shape: (N2, 4, 4)

    # 存储所有配对及其差异度
    pairs_diff = []
    for i, M1 in enumerate(poses1):
        for j, M2 in enumerate(poses2):
            M1_delta = delta_RT @ M1
            diff = measure_extrinsic_difference(M1_delta, M2)
            pairs_diff.append((i, j, diff))

    # 根据差异度升序排序
    pairs_diff.sort(key=lambda x: x[2])
    # 返回差异度最小的 K 个配对
    # 若总配对数小于 K，则只返回全部配对
    return pairs_diff[:min(K, len(pairs_diff))]


if __name__ == '__main__':
    npz_path1, npz_path2 = "//wsl.localhost/Ubuntu-22.04/home/aoki/honkai_neus/public_data/bunny_pose1/cameras_sphere.npz",\
        "//wsl.localhost/Ubuntu-22.04/home/aoki/honkai_neus/public_data/bunny_pose2/cameras_sphere.npz"
    npz_path1, npz_path2 = "/home/aoki/honkai_neus/public_data/keli_iphone_pos2/cameras_sphere.npz",\
        "/home/aoki/honkai_neus/public_data/keli_iphone_pos1/cameras_sphere.npz"
    delta_RT = np.array(
[[0.26563611, -0.45560444, 0.84962465, -0.10947776],
 [0.12074841, 0.89006960, 0.43954059, -0.06454727],
 [-0.95648172, -0.01416702, 0.29144814, 0.09227381],
 [0.00000000, 0.00000000, 0.00000000, 1.00000000]]
) # a test for obb based exc finding
    res = find_near_camera_poses(npz_path1, npz_path2, delta_RT)
    print(res)