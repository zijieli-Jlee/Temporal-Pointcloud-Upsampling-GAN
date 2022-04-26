#!/usr/bin/env python3
"""This script generates random fluid sequences with SPlisHSPlasH."""
import os
import re
import argparse
from copy import deepcopy
import sys
import json
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation

from glob import glob
import time
import tempfile
import subprocess
from shutil import copyfile
import itertools

import open3d as o3d
from physics_data_helper import numpy_from_bgeo, write_bgeo_from_numpy
from splishsplash_config import SIMULATOR_BIN, VOLUME_SAMPLING_BIN
import pysplishsplash as sph
import pysplishsplash.Utilities.SceneLoaderStructs as Scenes

SCRIPT_DIR = os.path.dirname(__file__)

# some constants for creating the objects in the simulation
# PARTICLE_RADIUS = 0.025
MAX_FLUID_START_VELOCITY_XZ = 2.0
MAX_FLUID_START_VELOCITY_Y = 0.5

MAX_RIGID_START_VELOCITY_XZ = 2.0
MAX_RIGID_START_VELOCITY_Y = 2.0

# default parameters for simulation
default_configuration = {
    "pause": False,
    "stopAt": 5.0,
    "particleRadius": -1,                 # need to be manually set
    "numberOfStepsPerRenderUpdate": 1,
    "density0": 1000,
    "simulationMethod": 4,
    "gravitation": [0, -9.81, 0],
    "cflMethod": 0,
    "cflFactor": 1,
    "cflMaxTimeStepSize": 0.005,
    "maxIterations": 100,
    "maxError": 0.01,
    "maxIterationsV": 100,
    "maxErrorV": 0.1,
    "stiffness": 50000,
    "exponent": 7,
    "velocityUpdateMethod": 0,
    "enableDivergenceSolver": True,
    "enablePartioExport": True,
    "enableRigidBodyExport": True,
    "dataExportFPS": 40,
    "partioAttributes": "density;velocity"
}

default_simulation = {
    "contactTolerance": 0.0125,
}

default_fluid = {
    "surfaceTension": 0.2,
    "surfaceTensionMethod": 0,
    "viscosity": 0.01,
    "viscosityMethod": 3,
    "viscoMaxIter": 200,
    "viscoMaxError": 0.05
}

default_rigidbody = {
    "translation": [0, 0, 0],
    "rotationAxis": [0, 1, 0],
    "rotationAngle": 0,
    "scale": [1.0, 1.0, 1.0],
    "color": [0.1, 0.4, 0.6, 1.0],
    "isDynamic": False,
    "isWall": True,
    "restitution": 0.6,
    "friction": 0.0,
    "collisionObjectType": 5,
    "collisionObjectScale": [1.0, 1.0, 1.0],
    "mapInvert": True,
}

default_fluidmodel = {"translation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}


def random_rotation_matrix(strength=None, dtype=None):
    """Generates a random rotation matrix 
    
    strength: scalar in [0,1]. 1 generates fully random rotations. 0 generates the identity. Default is 1.
    dtype: output dtype. Default is np.float32
    """
    if strength is None:
        strength = 1.0

    if dtype is None:
        dtype = np.float32

    x = np.random.rand(3)
    theta = x[0] * 2 * np.pi * strength
    phi = x[1] * 2 * np.pi
    z = x[2] * strength

    r = np.sqrt(z)
    V = np.array([np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z)])

    st = np.sin(theta)
    ct = np.cos(theta)

    Rz = np.array([[ct, st, 0], [-st, ct, 0], [0, 0, 1]])

    rand_R = (np.outer(V, V) - np.eye(3)).dot(Rz)
    return rand_R.astype(dtype)


def obj_volume_to_particles(objpath, radius, scale=1):
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, 'out.bgeo')
        scale_str = '{0}'.format(scale)
        radius_str = str(radius)
        status = subprocess.run([
            VOLUME_SAMPLING_BIN, '-i', objpath, '-o', outpath, '-r', radius_str,
            '-s', scale_str
        ])
        return numpy_from_bgeo(outpath)


def obj_surface_to_particles(objpath, radius):
    # currently required explicitly set radius
    # if radius is None:
    #     radius = PARTICLE_RADIUS
    obj = o3d.io.read_triangle_mesh(objpath)
    particle_area = np.pi * radius**2
    # 1.9 to roughly match the number of points of SPlisHSPlasHs surface sampling
    num_points = int(1.9 * obj.get_surface_area() / particle_area)
    pcd = obj.sample_points_poisson_disk(num_points, use_triangle_normal=True)
    points = np.asarray(pcd.points).astype(np.float32)
    normals = -np.asarray(pcd.normals).astype(np.float32)
    return points, normals


def rasterize_points(points, voxel_size, particle_radius):
    if not (voxel_size > 2 * particle_radius):
        raise ValueError(
            "voxel_size > 2*particle_radius is not true. {} > 2*{}".format(
                voxel_size, particle_radius))

    points_min = (points - particle_radius).min(axis=0)
    points_max = (points + particle_radius).max(axis=0)

    arr_min = np.floor_divide(points_min, voxel_size).astype(np.int32)
    arr_max = np.floor_divide(points_max, voxel_size).astype(np.int32) + 1

    arr_size = arr_max - arr_min

    arr = np.zeros(arr_size)

    offsets = []
    for z in range(-1, 2, 2):
        for y in range(-1, 2, 2):
            for x in range(-1, 2, 2):
                offsets.append(
                    np.array([
                        z * particle_radius, y * particle_radius,
                        x * particle_radius
                    ]))

    for offset in offsets:
        idx = np.floor_divide(points + offset, voxel_size).astype(
            np.int32) - arr_min
        arr[idx[:, 0], idx[:, 1], idx[:, 2]] = 1

    return arr_min, voxel_size, arr


def find_valid_fluid_start_positions(box_rasterized, fluid_rasterized):
    """Tries to find a valid starting position using the rasterized free space and fluid"""
    fluid_shape = np.array(fluid_rasterized[2].shape)
    box_shape = np.array(box_rasterized[2].shape)
    last_pos = box_shape - fluid_shape

    valid_fluid_start_positions_arr = np.zeros(box_shape)
    for idx in itertools.product(range(0, last_pos[0] + 1),
                                 range(0, last_pos[1] + 1),
                                 range(0, last_pos[2] + 1)):
        pos = np.array(idx, np.int32)
        pos2 = pos + fluid_shape
        view = box_rasterized[2][pos[0]:pos2[0], pos[1]:pos2[1], pos[2]:pos2[2]]
        if np.alltrue(
                np.logical_and(view, fluid_rasterized[2]) ==
                fluid_rasterized[2]):
            if idx[1] == 0:
                valid_fluid_start_positions_arr[idx[0], idx[1], idx[2]] = 1
            elif np.count_nonzero(
                    valid_fluid_start_positions_arr[idx[0], 0:idx[1], idx[2]]
            ) == 0:
                valid_fluid_start_positions_arr[idx[0], idx[1], idx[2]] = 1

    valid_pos = np.stack(np.nonzero(valid_fluid_start_positions_arr), axis=-1)
    selected_pos = valid_pos[np.random.randint(0, valid_pos.shape[0])]

    # update the rasterized bounding box volume by substracting the fluid volume
    pos = selected_pos
    pos2 = pos + fluid_shape
    view = box_rasterized[2][pos[0]:pos2[0], pos[1]:pos2[1], pos[2]:pos2[2]]
    box_rasterized[2][pos[0]:pos2[0], pos[1]:pos2[1], pos[2]:
                      pos2[2]] = np.logical_and(
                          np.logical_not(fluid_rasterized[2]), view)

    selected_pos += box_rasterized[0]
    selected_pos = selected_pos.astype(np.float) * box_rasterized[1]

    return selected_pos


def run_simulator(scene, output_dir):
    """Runs the simulator for the specified scene file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = sph.Exec.SimulatorBase()
        base.init(sceneFile=scene, useGui=False, useCache=False, initialPause=False, outputDir=output_dir)
        base.run()


def create_fluid_data(output_dir, seed, options):
    """Creates a random scene for a specific seed and runs the simulator"""

    np.random.seed(seed)

    bounding_boxes = sorted(glob(os.path.join(SCRIPT_DIR, 'models',
                                              'Box*.obj')))
    # override bounding boxes
    if options.default_box:
        bounding_boxes = [os.path.join(SCRIPT_DIR, 'models', 'Box.obj')]
    fluid_shapes = sorted(glob(os.path.join(SCRIPT_DIR, 'models',
                                            'Fluid*.obj')))
    rigid_shapes = sorted(
        glob(os.path.join(SCRIPT_DIR, 'models', 'RigidBody*.obj')))

    num_objects = np.random.choice([1, 2, 3])
    # override the number of objects to generate
    if options.num_objects > 0: num_objects = options.num_objects
    # num_objects = random.choice([1])
    print('num_objects', num_objects)

    particle_radius = options.particle_radius
    default_configuration['particleRadius'] = particle_radius
    print('particle_radius', particle_radius)

    # create fluids and place them randomly
    def create_fluid_object(radius):
        fluid_obj = np.random.choice(fluid_shapes)
        fluid = obj_volume_to_particles(fluid_obj, radius,
                                        scale=np.random.uniform(0.5, 1.5))[0]
        R = random_rotation_matrix(1.0)
        fluid = fluid @ R

        fluid_rasterized = rasterize_points(fluid, 2.01 * radius,
                                            radius)

        selected_pos = find_valid_fluid_start_positions(bb_rasterized,
                                                        fluid_rasterized)
        fluid_pos = selected_pos - fluid_rasterized[0] * fluid_rasterized[1]
        fluid += fluid_pos

        fluid_vel = np.zeros_like(fluid)
        max_vel = MAX_FLUID_START_VELOCITY_XZ
        fluid_vel[:, 0] = np.random.uniform(-max_vel, max_vel)
        fluid_vel[:, 2] = np.random.uniform(-max_vel, max_vel)
        max_vel = MAX_FLUID_START_VELOCITY_Y
        fluid_vel[:, 1] = np.random.uniform(-max_vel, max_vel)

        density = np.random.uniform(500, 2000)
        viscosity = np.random.exponential(scale=1 / 20) + 0.01
        if options.uniform_viscosity:
            viscosity = np.random.uniform(0.01, 0.3)
        elif options.log10_uniform_viscosity:
            viscosity = 0.01 * 10**np.random.uniform(0.0, 1.5)

        if options.default_density: density = 1000
        if options.default_viscosity: viscosity = 0.01

        return {
            'type': 'fluid',
            'positions': fluid,
            'velocities': fluid_vel,
            'density': density,
            'viscosity': viscosity,
        }

    scene_is_valid = False
    create_coarse = False

    if options.coarse_ratio != 1.0:
        if options.coarse_ratio < 1.0:
            raise Exception(f'Coarse ratio must be smaller than 1.0 (current: {options.coarse_ratio})')
        create_coarse = True

    for create_scene_i in range(100):
        if scene_is_valid:
            break

        # select random bounding box
        bb_obj = np.random.choice(bounding_boxes)
        # convert bounding box to particles
        bb, bb_normals = obj_surface_to_particles(bb_obj, particle_radius)
        bb_vol = obj_volume_to_particles(bb_obj, particle_radius)[0]

        # rasterize free volume
        bb_rasterized = rasterize_points(np.concatenate([bb_vol, bb], axis=0),
                                         2.01 * particle_radius,
                                         particle_radius)
        bb_rasterized = bb_rasterized[0], bb_rasterized[1], binary_erosion(
            bb_rasterized[2], structure=np.ones((3, 3, 3)), iterations=3)
        if create_coarse:
            print('Will create another coarse scene')
            coarse_particle_radius = particle_radius*options.coarse_ratio
            c_bb, c_bb_normals = obj_surface_to_particles(bb_obj, coarse_particle_radius)
            c_bb_vol = obj_volume_to_particles(bb_obj, coarse_particle_radius)[0]

            # rasterize free volume
            c_bb_rasterized = rasterize_points(np.concatenate([c_bb_vol, c_bb], axis=0),
                                             2.01 * coarse_particle_radius,
                                             coarse_particle_radius)
            c_bb_rasterized = c_bb_rasterized[0], c_bb_rasterized[1], binary_erosion(
                c_bb_rasterized[2], structure=np.ones((3, 3, 3)), iterations=3)
        objects = []
        if create_coarse:
            coarse_objects = []

        create_fn_list = [create_fluid_object]

        for object_i in range(num_objects):

            create_fn = np.random.choice(create_fn_list)

            create_success = False
            for i in range(10):
                if create_success:
                    break
                try:
                    obj = create_fn(particle_radius)
                    objects.append(obj)
                    create_success = True
                    print('create object success')
                    if create_coarse:
                        c_obj = create_fn(coarse_particle_radius)
                        coarse_objects.append(c_obj)
                except:
                    print('create object failed')
                    pass

        scene_is_valid = True

        def get_total_number_of_fluid_particles(obj_lst):
            num_particles = 0
            for obj in obj_lst:
                if obj['type'] == 'fluid':
                    num_particles += obj['positions'].shape[0]
            return num_particles

        def get_smallest_fluid_object(obj_lst):
            num_particles = 100000000
            obj_idx = -1
            for idx, obj in enumerate(obj_lst):
                if obj['type'] == 'fluid':
                    if obj['positions'].shape[0] < num_particles:
                        obj_idx = idx
                    num_particles = min(obj['positions'].shape[0],
                                        num_particles)
            return obj_idx, num_particles

        total_number_of_fluid_particles = get_total_number_of_fluid_particles(objects)

        if options.const_fluid_particles:
            if options.const_fluid_particles > total_number_of_fluid_particles:
                scene_is_valid = False
            else:
                while get_total_number_of_fluid_particles(objects) != options.const_fluid_particles:
                    difference = get_total_number_of_fluid_particles(objects
                    ) - options.const_fluid_particles
                    obj_idx, num_particles = get_smallest_fluid_object(objects)
                    if num_particles < difference:
                        del objects[obj_idx]
                    else:
                        objects[obj_idx]['positions'] = objects[obj_idx][
                            'positions'][:-difference]
                        objects[obj_idx]['velocities'] = objects[obj_idx][
                            'velocities'][:-difference]

        if options.max_fluid_particles:
            if options.max_fluid_particles < total_number_of_fluid_particles:
                scene_is_valid = False

    sim_directory = os.path.join(output_dir, 'sim_{0:04d}'.format(seed))
    os.makedirs(sim_directory, exist_ok=False)
    # generate scene json file
    scene = {
        'Configuration': default_configuration,
        'Simulation': default_simulation,
        # 'Fluid': default_fluid,
        'RigidBodies': [],
        'FluidModels': [],
    }

    rigid_body_next_id = 1

    # bounding box
    box_output_path = os.path.join(sim_directory, 'box.bgeo')
    write_bgeo_from_numpy(box_output_path, bb, bb_normals)

    box_obj_output_path = os.path.join(sim_directory, 'box.obj')
    copyfile(bb_obj, box_obj_output_path)

    rigid_body = deepcopy(default_rigidbody)
    rigid_body['id'] = rigid_body_next_id
    rigid_body_next_id += 1
    rigid_body['geometryFile'] = os.path.basename(
        os.path.abspath(box_obj_output_path))
    rigid_body['mapResolution'] = [64, 64, 64]
    rigid_body["collisionObjectType"] = 5
    scene['RigidBodies'].append(rigid_body)

    fluid_count = 0
    for obj in objects:
        fluid_id = 'fluid{0}'.format(fluid_count)
        fluid_count += 1
        fluid = deepcopy(default_fluid)
        fluid['viscosity'] = obj['viscosity']
        fluid['density0'] = obj['density']
        scene[fluid_id] = fluid

        fluid_model = deepcopy(default_fluidmodel)
        fluid_model['id'] = fluid_id

        fluid_output_path = os.path.join(sim_directory, fluid_id + '.bgeo')
        write_bgeo_from_numpy(fluid_output_path, obj['positions'],
                              obj['velocities'])
        fluid_model['particleFile'] = os.path.basename(fluid_output_path)
        scene['FluidModels'].append(fluid_model)

    scene_output_path = os.path.join(sim_directory, 'scene.json')

    with open(scene_output_path, 'w') as f:
        json.dump(scene, f, indent=4)

    if create_coarse:
        coarse_config = default_configuration
        coarse_config['cflMaxTimeStepSize'] = options.coarse_ratio * coarse_config['cflMaxTimeStepSize']
        coarse_config['particleRadius'] = options.coarse_ratio * coarse_config['particleRadius']
        coarse_scene = {
            'Configuration': coarse_config,
            'Simulation': default_simulation,
            # 'Fluid': default_fluid,
            'RigidBodies': [],
            'FluidModels': [],
        }
        coarse_sim_directory = os.path.join(output_dir, 'sim_{0:04d}_coarse'.format(seed))
        os.makedirs(coarse_sim_directory, exist_ok=False)

        c_rigid_body_next_id = 1

        # bounding box
        c_box_output_path = os.path.join(coarse_sim_directory, 'box.bgeo')
        write_bgeo_from_numpy(c_box_output_path, c_bb, c_bb_normals)

        c_box_obj_output_path = os.path.join(coarse_sim_directory, 'box.obj')
        copyfile(bb_obj, c_box_obj_output_path)

        rigid_body = deepcopy(default_rigidbody)
        rigid_body['id'] = c_rigid_body_next_id
        rigid_body_next_id += 1
        rigid_body['geometryFile'] = os.path.basename(
            os.path.abspath(c_box_obj_output_path))
        rigid_body['mapResolution'] = [64, 64, 64]
        rigid_body["collisionObjectType"] = 5
        coarse_scene['RigidBodies'].append(rigid_body)

        fluid_count = 0
        for c_obj in coarse_objects:
            fluid_id = 'fluid{0}'.format(fluid_count)
            fluid_count += 1
            fluid = deepcopy(default_fluid)
            fluid['viscosity'] = c_obj['viscosity']
            fluid['density0'] = c_obj['density']
            coarse_scene[fluid_id] = fluid

            fluid_model = deepcopy(default_fluidmodel)
            fluid_model['id'] = fluid_id

            fluid_output_path = os.path.join(coarse_sim_directory, fluid_id + '.bgeo')
            write_bgeo_from_numpy(fluid_output_path, c_obj['positions'],
                                  c_obj['velocities'])
            fluid_model['particleFile'] = os.path.basename(fluid_output_path)
            coarse_scene['FluidModels'].append(fluid_model)

        coarse_scene_output_path = os.path.join(coarse_sim_directory, 'scene.json')

        with open(coarse_scene_output_path, 'w') as f:
            json.dump(coarse_scene, f, indent=4)
    if options.run_sim:
        run_simulator(os.path.abspath(scene_output_path), sim_directory)
        if create_coarse:
            run_simulator(os.path.abspath(coarse_scene_output_path), coarse_sim_directory)


def run_with_configs(sim_directory):
    scene_path = os.path.join(sim_directory, 'scene.json')
    run_simulator(os.path.abspath(scene_path), sim_directory)


def main():
    parser = argparse.ArgumentParser(description="Creates physics sim sim_data")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="The path to the output directory")
    parser.add_argument("--seed",
                        type=int,
                        required=True,
                        help="The random seed for initialization")
    parser.add_argument(
        "--uniform-viscosity",
        action='store_true',
        help="Generate a random viscosity value from a uniform distribution")
    parser.add_argument(
        "--log10-uniform-viscosity",
        action='store_true',
        help=
        "Generate a random viscosity value from a uniform distribution in log10 space"
    )
    parser.add_argument(
        "--default-viscosity",
        action='store_true',
        help="Force all generated fluids to have the default viscosity")
    parser.add_argument(
        "--default-density",
        action='store_true',
        help="Force all generated objects to have the default density")
    parser.add_argument(
        "--default-box",
        action='store_true',
        help="Force all generated scenes to use the default bounding box")
    parser.add_argument(
        "--num-objects",
        type=int,
        default=0,
        help=
        "The number of objects to place in the scene. 0 (default value) means random choice from 1 to 3"
    )
    parser.add_argument(
        "--const-fluid-particles",
        type=int,
        default=0,
        help="If set a constant number of particles will be generated.")
    parser.add_argument(
        "--max-fluid-particles",
        type=int,
        default=0,
        help="If set the number of particles will be limited.")
    parser.add_argument(
        "--run_with_config",
        action='store_true',
        help="Run a simulation with cached scene file")
    parser.add_argument(
        "--run_sim",
        action='store_true',
        help="Run a simulation after creating scenes")
    parser.add_argument(
        "--particle_radius",
        type=float,
        default=0.025,
        help=
        "Radius of particles"
    )
    parser.add_argument(
        "--coarse_ratio",
        type=float,
        default=1.0,
        help=
        "If needed to run another coarse simulation on the same scene"
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if args.run_with_config:
        sim_directory = os.path.join(args.output, 'sim_{0:04d}'.format(args.seed))
        run_with_configs(sim_directory)  # there should already be a scene file inside
    else:
        create_fluid_data(args.output, args.seed, args)


if __name__ == '__main__':
    sys.exit(main())
