# Training and Inference pipeline for Human Activity prediction

Project based on Pytorch-Lightning

### Training
Experiments are written in Yaml format and stored in experiments folder

`EXPERIMENT_NAME: "baseline"` - You need to specify experiment name, this name will be used to store logs

`DATA: ...` - This block is used to specify what samples to train on and where the data stored. Moreover, `BATCH_SIZE` and validation size are needed to be specified

`OPTIMIZER: ...` - Optimizer configuration: from what package import, what optimizer to use and arguments to this optimizer
