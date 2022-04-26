import json
import numpy as np
from physics_data_helper import *


def dump_data(data, out_dir, frame_idx):
    np.savez(os.path.join(out_dir, 'data_{}.npz'.format(frame_idx)), **data)


def create_scene_files(scene_dir, out_dir):

    with open(os.path.join(scene_dir, 'scene.json'), 'r') as f:
        scene_dict = json.load(f)

    box, box_normals = numpy_from_bgeo(os.path.join(scene_dir, 'box.bgeo'))

    partio_dir = os.path.join(scene_dir, 'partio')
    fluid_ids = get_fluid_ids_from_partio_dir(partio_dir)
    num_fluids = len(fluid_ids)
    fluid_id_bgeo_map = {
        k: get_fluid_bgeo_files(partio_dir, k) for k in fluid_ids
    }

    frames = None

    for k, v in fluid_id_bgeo_map.items():
        if frames is None:
            frames = len(v)
        if len(v) != frames:
            raise Exception(
                'number of frames for fluid {} ({}) is different from {}'.
                format(k, len(v), frames))

    for frame_i in range(frames):

        # only save the box for the first frame of each file to save memory
        # if frame_i == 0:
        #     boundary_feat_dict = {}
        #     boundary_feat_dict['box'] = box.astype(np.float32)
        #     boundary_feat_dict['box_normals'] = box_normals.astype(np.float32)
        #     box_path = os.path.join(out_dir, 'rigid_body.npz')
        #     np.savez(box_path, **boundary_feat_dict)

        sizes = np.array([0, 0, 0, 0], dtype=np.int32)
        feat_dict = {}
        pos = []
        vel = []
        viscosity = []

        for flid in fluid_ids:
            bgeo_path = fluid_id_bgeo_map[flid][frame_i]
            pos_, vel_ = numpy_from_bgeo(bgeo_path)
            pos.append(pos_)
            vel.append(vel_)
            # prs.append(prs_)

            viscosity.append(
                        np.full(pos_.shape[0:1],
                                scene_dict[flid]['viscosity'],
                                dtype=np.float32))

            # only needed when simulating different types of fluids
            # mass.append(
            #     np.full(pos_.shape[0:1],
            #             scene_dict[flid]['density0'],
            #             dtype=np.float32))

            sizes[0] += pos_.shape[0]

        pos = np.concatenate(pos, axis=0)
        vel = np.concatenate(vel, axis=0)
        # prs = np.concatenate(prs, axis=0)
        viscosity = np.concatenate(viscosity, axis=0)

        feat_dict['pos'] = pos.astype(np.float32)
        feat_dict['vel'] = vel.astype(np.float32)
        # feat_dict['prs'] = prs.astype(np.float32)
        # feat_dict['viscosity'] = viscosity.astype(np.float32)
        dump_data(feat_dict, out_dir, frame_i)


if __name__ == '__main__':
    os.mkdir('./train_data_0.025_fine')
    for i in range(1, 21):
        print(f'Dumping case {i}')
        scene_dir = os.path.join('./train_data_0.025_fine', 'sim_{0:04d}'.format(i))
        os.mkdir(f'./train_data_0.025_fine/case{i}')
        create_scene_files(scene_dir, f'./train_data_0.025_fine/case{i}')

    os.mkdir('./test_data_0.025_fine')
    for i in range(1, 5):
        print(f'Dumping case {i}')
        scene_dir = os.path.join('./test_data_0.025_raw', 'sim_{0:04d}'.format(i))
        os.mkdir(f'./test_data_0.025_fine/case{i}')
        create_scene_files(scene_dir, f'./test_data_0.025_fine/case{i}')
