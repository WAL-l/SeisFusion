# SeisFusion
this is the code implementation of "SeisFusion: Constrained Diffusion Model with Input Guidance for 3D Seismic Data Interpolation and Reconstruction"

# Train

## Installation
run:
```
pip install -r requirements.txt
```
## Training models

### Preparing Data

The training code reads npy data from a directory of data files. 

For creating your own dataset, simply dump all of your datas into a directory with ".npy" extensions. 

Simply pass --data_dir path/to/datas to the training script, and it will take care of the rest.

## training
To train your model, you should first decide some hyperparameters. We will split up our hyperparameters into three groups: model architecture, diffusion process, and training flags. Here are some reasonable defaults for a baseline:
```
MODEL_FLAGS="--data_size 256 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-5 --batch_size 4"
```
Note that you should change the size [here](https://github.com/WAL-l/Reconstruction/blob/65033b6f3ea7540f4fee675e91c1e6cc49d73973/train/guided_diffusion/train_util.py#L160) to your own data size.
Once you have setup your hyper-parameters, you can run an experiment like so:
```
python train.py --data_dir path/to/datas $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```
You may also want to train in a distributed manner. In this case, run the same command with mpiexec:
```
mpiexec -n $NUM_GPUS python train.py --data_dir path/to/datas $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```
When training in a distributed manner, you must manually divide the --batch_size argument by the number of ranks. In lieu of distributed training, you may use --microbatch 16 (or --microbatch 1 in extreme memory-limited cases) to reduce memory usage.

The logs and saved models will be written to a logging directory determined by the OPENAI_LOGDIR environment variable. If it is not set, then a temporary directory will be created in /tmp.

## Reconstruction
```bash
python reconstruction.py --conf_path confs/reconstruction.yml
```
Note that you should change the size [here](https://github.com/WAL-l/SeisFusion/blob/536e90667678c5c205dbeae6f143747ce797dd43/reconstruction.py#L132) to your own data size.
The reconstruction.yml contains information about your diffusion model and specifies the location of the weights file, the location of the file to be reconstructed, and the location of the mask file.
Find the output in `./log`

## Details on data



**How to prepare the test data?**

one file for the data which is ".npy"
one file for mask
The masks have the value 1 for known regions and 0 for unknown areas.