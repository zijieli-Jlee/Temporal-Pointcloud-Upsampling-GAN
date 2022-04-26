### Training
For action data as the underlying points is relatively uniform, therefore it is advised to train a non-masking version of the model.
We provide the pretrained model and bash command to train the model under the folder ```train_dir```.

### Generation
In the ```demo.ipynb```, we provide an example on how to use trained model to generate high-resolution reconstruction sequences from input.

### Feature evaluation
We can use the output of the last Flow Embedding layer as features to train a simple classification model (comprises a Set Abstraction layer and a MLP).
This can be done by loading the trained ckpt to ```eval_temp_feat.py```, we provide an example command in the ```eval_dis``` folder. The training code used for classification is modified from [PointNet2Pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/train_classification.py) and [PST-Conv](https://github.com/hehefan/Point-Spatio-Temporal-Convolution/blob/main/train-msr.py)'s training code.
