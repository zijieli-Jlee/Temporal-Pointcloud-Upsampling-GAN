import numpy as np
import numba as nb
import open3d as o3d
import torch
from scipy.spatial import KDTree
from dgl.geometry import farthest_point_sampler
import frnn
from typing import Union
from sampling import farthest_point_sampling as fps
BASE_RADIUS = 0.025


def voxel_downsample(pos, radius, ds_ratio):

    if not isinstance(pos, np.ndarray):
        pos = pos.numpy()

    if pos.shape[1] != 3:
        pos = pos.reshape(-1, 3)

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pos)
    pcd1 = pcd0.voxel_down_sample((1./ds_ratio)*radius+1e-9)
    ds_pos = np.asarray(pcd1.points).astype(np.float32)

    target_size = int(ds_ratio*pos.shape[0])
    if ds_pos.shape[0] > target_size:
        slice = np.random.choice(ds_pos.shape[0], target_size, replace=False)
        ds_pos = ds_pos[slice]
    return ds_pos


def sample_patch(input_pos: np.ndarray, h:  float, return_free_surface_particles=True):
    total_num = input_pos.shape[0]      # total particle number
    if total_num > 80000:
        patch_num = 32768
    elif total_num > 40000:
        patch_num = 16384
    elif total_num > 10000:
        patch_num = 8192
    else:
        patch_num = total_num
    tree = KDTree(input_pos)
    sampling_count = 1
    ds_sample_num = 0
    while ds_sample_num < 500:
        idx = np.random.choice(total_num)
        start_point = input_pos[idx]
        _, patch = tree.query(start_point, patch_num)
        patch_pos = input_pos[patch]

        sampling_count += 1
        ds_pos = voxel_downsample(patch_pos, radius=BASE_RADIUS / h, ds_ratio=0.50)
        ds_sample_num = ds_pos.shape[0]

        if sampling_count > 100:

            visualize_pointcloud(input_pos)
            visualize_pointcloud(patch_pos)
            visualize_pointcloud(ds_pos)
            raise Exception('Abnormal sampling times!')
    if return_free_surface_particles:
        surface_points = get_free_surface_particles(patch_pos, 2.2*BASE_RADIUS / h)
        return patch_pos, ds_pos, surface_points
    return patch_pos, ds_pos


def voxel_downsample_with_feat(pos, feat, radius, ds_ratio):
    if not isinstance(pos, np.ndarray):
        pos = pos.numpy()
    if not isinstance(feat, np.ndarray):
        feat = feat.numpy()
    feat = np.concatenate((feat, np.zeros((1, feat.shape[1]))), axis=0)

    if pos.shape[1] != 3:
        pos = pos.reshape(-1, 3)

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pos)
    [pcd1, trace, _] = pcd0.voxel_down_sample_and_trace((1./ds_ratio)*radius+1e-9,
                                                        min_bound=pcd0.get_min_bound(),
                                                        max_bound=pcd0.get_max_bound()
                                                        )
    ds_pos = np.asarray(pcd1.points).astype(np.float32)
    trace_idx = np.asarray(trace)
    p_in_voxel = np.sum(trace_idx != -1, axis=-1, keepdims=True)
    feat_in_voxel = feat[trace_idx]
    ds_feat = np.sum(feat_in_voxel, axis=1) / p_in_voxel

    target_size = int(ds_ratio*pos.shape[0])
    if ds_pos.shape[0] > target_size:
        slice = np.random.choice(ds_pos.shape[0], target_size, replace=False)
        ds_pos = ds_pos[slice]
        ds_feat = ds_feat[slice]
    return ds_pos, ds_feat.astype(np.float32)


def sample_patch_with_fps(input_pos: np.ndarray,
                          h:  float,
                          sample_num=None,
                          return_free_surface_particles=True,
                          return_patch_and_fps_idx=False):
    # notice: this function also downsample feature
    # feature must be in same order as input_pos
    total_num = input_pos.shape[0]      # total particle number
    if sample_num is None:
        # when testing emd, number need to be k*1024
        if total_num > 10000:
            patch_num = 9216
        else:
            patch_num = int(total_num//1024)*1024
    else:
        if total_num > sample_num:
            patch_num = sample_num
        else:
            patch_num = 4096

    tree = KDTree(input_pos)

    idx = np.random.choice(total_num)
    start_point = input_pos[idx]
    _, patch = tree.query(start_point, patch_num, workers=1)
    patch_pos = input_pos[patch]

    # fps_idx, _ = fps(patch_pos, int((0.125 + np.random.uniform(0., 0.025)) * patch_num))
    fps_idx, _ = fps(patch_pos, int(0.125 * patch_num))

    return_dict = {
                    'patch_pos': patch_pos,
                    'ds_pos': patch_pos[fps_idx],
    }
    if return_free_surface_particles:
        surface_points = get_free_surface_particles(patch_pos, 3.1*BASE_RADIUS / h)
        return_dict['surface_points'] = surface_points

    if return_patch_and_fps_idx:
        return return_dict, patch, fps_idx

    return return_dict


def sample_patch_feat(input_pos: np.ndarray,
                      input_feat: np.ndarray,
                      h:  float,
                      return_next_frame_sample=False,
                      return_free_surface_particles=True,
                      next_frame_pos=None):
    # notice: this function also downsample feature
    # feature must be in same order as input_pos
    total_num = input_pos.shape[0]      # total particle number
    if total_num > 20000:
        patch_num = 20000  # to train siamese
    elif total_num > 10000:
        patch_num = 8192
    else:
        patch_num = total_num
    tree = KDTree(input_pos)
    sampling_count = 1
    ds_sample_num = 0
    while ds_sample_num < 500:
        idx = np.random.choice(total_num)
        start_point = input_pos[idx]
        _, patch = tree.query(start_point, patch_num)
        patch_pos = input_pos[patch]
        patch_feat = input_feat[patch]

        sampling_count += 1
        ds_pos, ds_feat = voxel_downsample_with_feat(patch_pos, patch_feat, BASE_RADIUS / h, ds_ratio=0.50)
        ds_sample_num = ds_pos.shape[0]

        if sampling_count > 100:

            visualize_pointcloud(input_pos)
            visualize_pointcloud(patch_pos)
            visualize_pointcloud(ds_pos)
            raise Exception('Abnormal sampling times!')
    return_dict = {
                    'patch_pos': patch_pos,
                    'patch_feat': patch_feat,
                    'ds_pos': ds_pos,
                    'ds_feat': ds_feat,
    }
    if return_free_surface_particles:
        surface_points = get_free_surface_particles(patch_pos, 2.2*BASE_RADIUS / h)
        return_dict['surface_points'] = surface_points

    if return_next_frame_sample and next_frame_pos is not None:
        next_frame_patch_pos = next_frame_pos[patch]
        next_ds_pos = voxel_downsample(next_frame_patch_pos, radius=BASE_RADIUS / h, ds_ratio=0.50)
        return_dict['next_patch_pos'] = next_frame_patch_pos
        return_dict['next_ds_pos'] = next_ds_pos

    return return_dict


def visualize_pointcloud(pos):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    o3d.visualization.draw_geometries([pcd])

def get_distribution_info(points):
    """
    return the centroid, min/max bound of point cloud
    """
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)

    centroid = np.mean(points, axis=0)
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    return centroid, min_bound, max_bound


def normalize_point_cloud(pcd_pos):
    centroid = np.mean(pcd_pos, axis=0, keepdims=True)
    input = pcd_pos - centroid
    # furthest_distance = np.amax(
    #     np.sqrt(np.sum(input ** 2, axis=-1, keepdims=True)), axis=0)
    furthest_distance = np.float32(1.)
    input /= furthest_distance
    return input, centroid, furthest_distance


def dump_pointcloud_visualization(pcd_pos, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # ctr = vis.get_view_control()
    # parameters = o3d.io.read_pinhole_camera_parameters(
    #     './ScreenCamera_2021-08-20-22-26-00.json')
    # ctr.convert_from_pinhole_camera_parameters(parameters)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_pos)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename)
    vis.destroy_window()


def filter_overlap_particles(pcd_pos: Union[torch.Tensor, np.ndarray], h=BASE_RADIUS*0.5) -> np.ndarray:
    # this function is not differentiable
    # no computational graph is created or retained
    if isinstance(pcd_pos, torch.Tensor):
        pcd_pos_np = pcd_pos.detach().cpu().numpy()
    else:
        pcd_pos_np = pcd_pos
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pcd_pos_np)
    [pcd1, _, _] = pcd0.voxel_down_sample_and_trace(h + 1e-8,
                                                         min_bound=pcd0.get_min_bound(),
                                                         max_bound=pcd0.get_max_bound()
                                                        )
    ds_pos = np.asarray(pcd1.points).astype(np.float32)
    return ds_pos


@nb.njit
def linear_kernel(r, re):
    if r > re:
        ker = 0.
    elif r < 1e-8:
        ker = 0.
    else:
        ker = re / r - 1.
    return ker


def fixed_radius_neighbor_num(pos, radius):
    tree = KDTree(pos)
    nbr_num = tree.query_ball_point(pos, radius, return_length=True)
    return np.asarray(nbr_num)


def fps_numpy_wrapper(pos, n_points):
    pos_tsr = torch.from_numpy(pos)
    sampled_idx = farthest_point_sampler(pos_tsr.unsqueeze(0), n_points)
    return sampled_idx[0].numpy()


def get_free_surface_particles(pos, radius):
    nbr_num = fixed_radius_neighbor_num(pos, radius)
    sorted_nbr_num = np.sort(nbr_num)
    threshold = np.mean(sorted_nbr_num[int(pos.shape[0] * 0.95):-int(pos.shape[0] * 0.01)])
    surface_points = pos[nbr_num < 0.85 * threshold]
    return surface_points


if __name__ == '__main__':
    import faiss
    test_data_path = 'data/train_data_0.025_fine/case1/data_100.npz'
    test_data = np.load(test_data_path)
    pos = test_data['pos']
    # dump_pointcloud_visualization(pos, 'test.png')
    nbr_num = fixed_radius_neighbor_num(pos, 0.055)

    sorted_nbr_num = np.sort(nbr_num)
    threshold = np.mean(sorted_nbr_num[int(pos.shape[0]*0.95):-int(pos.shape[0]*0.01)])
    surface_points = pos[nbr_num < 0.9*threshold]
    visualize_pointcloud(surface_points)
    fps_points = fps_numpy_wrapper(surface_points, 1024)
    visualize_pointcloud(surface_points[fps_points])

    # _, sample = patch_downsample(test_data)
    # print(pos.shape)
    # print(sample[0]['pos'].shape)
    # visualize_pointcloud(sample[0]['pos'])