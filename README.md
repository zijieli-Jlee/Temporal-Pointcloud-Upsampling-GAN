## Temporal-Point-Cloud-Upsampling-GAN
Code repo for TPU-GAN ([paper](https://openreview.net/pdf?id=FEBFJ98FKx))

### Requirement
* PyTorch3D from: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
* FRNN from: https://github.com/lxxue/FRNN
* Chamfer distance module: https://github.com/krrish94/chamferdist
* PointNet2 ops from: https://github.com/erikwijmans/Pointnet2_PyTorch (could be installed by cloning the repo and run ```pip install pointnet2_ops_lib/.```)
* Other packages can be installed through: ```conda env create -f environment.yml```

The code is tested under Linux Ubuntu 18.04 with CUDA 10.2.

### Dataset
We use DFSPH solver from SPLISHSPLASH (https://github.com/InteractiveComputerGraphics/SPlisHSPlasH) to generate simulation data. The geometry assets and Python script for simulation are taken and tweaked from CConv: https://github.com/isl-org/DeepLagrangianFluids. </br>

To generate fluid data, first install SPLISHSPLASH following: https://splishsplash.readthedocs.io/en/latest/build_from_source.html (with Python bindings turned on), and set the path to binary at line 3 of ```splishsplash_config.py``` under ```fluid_data_generation``` folder. 

Then run SPH simulation by:
```bash
cd fluid_data_generation
python sim_fluid_sequence.py
```
And process the simulation data (in bgeo format) into npz data. This step requires Partio (https://github.com/wdas/partio) installed.
```bash
python process_training_data.py
```
We also provide generated data at link: https://drive.google.com/drive/folders/1313m62z5mM_vEUg0ptFrnuZP8YJmwKQ2?usp=sharing.

The MSR-Action dataset is taken from MeteorNet: https://github.com/xingyul/meteornet/tree/master/action_cls.

### Training and inference
* The train and test data can be generated using generate_data.py in "./training" folder: 
 ```
 cd training
 python generate_data.py
 ``` 
* To run the simulation using pretrained GNN:
The example simulation script is in the "./running_script" folder. </br>
```
cd running_script
sh gnn_simulate.sh
``` 

### Visualization and rendering
We use Maya with [Partio plugin](https://github.com/InteractiveComputerGraphics/MayaPartioTools) to load and render the results (in .bgeo format).

### Acknowledgement
* Conjugate gradient solver in Pytorch: https://github.com/sbarratt/torch_cg
* Continuous convolution for end-to-end position based fluid simulation: https://github.com/isl-org/DeepLagrangianFluids
* SPH library: https://github.com/InteractiveComputerGraphics/SPlisHSPlasH
