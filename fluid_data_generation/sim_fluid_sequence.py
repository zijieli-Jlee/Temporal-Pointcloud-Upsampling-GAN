import os

# for training data
seq_num = 20
output_dir = 'train_data_0.025_raw/'
#
for seed in range(1, seq_num+1):
    os.system('python create_physics_scenes.py' +
              ' --output ' + output_dir +
              ' --seed ' + str(seed) +
              ' --default-viscosity' +
              ' --default-density' +
              ' --run_sim' +
              ' --particle_radius 0.0125'
              )


# for testing data
seq_num = 4
output_dir = 'test_data_0.025_raw/'
#
for seed in range(1, seq_num+1):
    os.system('python create_physics_scenes.py' +
              ' --output ' + output_dir +
              ' --seed ' + str(seed) +
              ' --default-viscosity' +
              ' --default-density' +
              ' --run_sim' +
              ' --particle_radius 0.0125'
              )
