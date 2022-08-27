# Training and Inference pipeline for Human Activity prediction

Project based on Pytorch-Lightning

### Training

Experiments are written in Yaml format and stored in experiments folder

`EXPERIMENT_NAME: "baseline"` - You need to specify experiment name, this name will be used to store logs

`DATA: ...` - This block is used to specify what samples to train on and where the data stored. Moreover, `BATCH_SIZE` and validation size are needed to be specified

`OPTIMIZER: ...` - Optimizer configuration: from what package import, what optimizer to use and arguments to this optimizer

To run training process You need to run `python main.py --experiment_cfg experiments/train.yaml`

`MODEL: ...` - What model to use

`SCHEDULER: ...` - Parameters of scheduler to use.

`CRITERION: ...` - What loss to use to optimize

`AUGMENTATION: ...` - The training augmentations. The augmentation need to be written in package specified in `PY`

`EARLY_STOPPING: ...` - Parameters of early stopping process

`CHECKPOINT: ...` - Parameters of checkpointing process

`EPOCHS: ...` - The number of epochs

`GPUS: ...` - GPUS to train on


To start training process with baseline experiment config run : 

`python main.py --experiment_cfg experiments/baseline.yaml`


###Convert checkpoint of Pytorch-Lightning

To convert checkpoint of Pytorch-Lightning run: 

`python convert_checkpoint.py --checkpoint path_to_checkpoint --config_file path_to_config`

###Inference

Inference config has the following structure:

`MODEL: ...` - What model to use

`TEST: ...` - This block is used to specify what samples to test on and where the data stored


To run inference 
 `python inference.py --config_file test.yaml`

