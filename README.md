# Pose-Assisted Multi-Camera Collaboration for Active Object Tracking
This repository is the python implementation of [Pose-Assisted Multi-Camera Collaboration for Active Object Tracking (AAAI 2020)](https://arxiv.org/abs/2001.05161).

It contains the code for training/testing(Pytorch). The 3D environments are hosted in [gym-unrealcv](https://github.com/zfw1226/gym_unrealcv).

![Task](https://github.com/LilJing/pose-assisted-collaboration/blob/master/images/task.jpg)

## Dependencies
The repository requires:
- Linux (Ubuntu 16)
- Python >= 3.6
- Pytorch >= 1.0
- [gym-unrealcv](https://github.com/zfw1226/gym-unrealcv) >= 1.0
- OpenCV >= 3.4
- Numpy == 1.14.0
- setproctitle, scikit-image, imageio, TensorboardX, Matplotlib

## Prepare 3D Environments
Install [gym-unrealcv](https://github.com/zfw1226/gym_unrealcv):
```bash
git clone https://github.com/zfw1226/gym_unrealcv
cd gym_unrealcv
pip install -e .
```
Load environment binaries:
```bash
python load_env.py -e Textures
python load_env.py -e MCRoom
python load_env.py -e UrbanTree
python load_env.py -e Garedn
```

## Installation

To download the repository and install the requirements, you can run as:
```
git clone https://github.com/LilJing/pose-assisted-collaboration.git
cd pose-assisted-collaboration
pip install -r requirements.txt
```
Note that you need install `OpenCV`, `Pytorch`, and the `3D environments` additionally.
## Training
### Train the vision-based controller
Use the following command:
```
python main.py --rescale --shared-optimizer --env UnrealMCRoom-DiscreteColorGoal-v5 --workers 6
```
### Train the pose-based controller
```
cd ./pose
python main.py --env PoseEnv-v1 --shared-optimizer --workers 12
```
The best parameters of the network will be saved in corresponding `logs` dir.

## Evaluation

There are two environments for evaluation, _Garden_ and _Urban City_.

We provide the pre-trained model in `.models/`.
The trained vision-based controller model is `Vision-model-best.dat` and the pose-based controller model is `Pose-model-best.dat`.

Run our model on _Garden_:
```
python evaluate.py --rescale --load-vision-model  ./models/Vision-model-best.dat --load-pose-model ./models/Pose-model-best.dat  --env UnrealGarden-DiscreteColorGoal-v1 --num-episodes 100 --test-type modelgate --render
```
Run our model on _Urban City_:
```
python evaluate.py --rescale --load-vision-model  ./models/Vision-model-best.dat --load-pose-model ./models/Pose-model-best.dat  --env UnrealUrbanTree-DiscreteColorGoal-v1 --num-episodes 100 --test-type modelgate --render
```

## Demo Videos

To see demo videos, please refer to [YouTube](https://www.youtube.com/watch?v=8Ha7HGkRv6k&feature=youtu.be).

## Citation
If you found this work useful, please consider citing:
```
@inproceedings{li2020pose,
  title={Pose-Assisted Multi-Camera Collaboration for Active Object Tracking},
  author={Jing Li, "Jing Xu, Fangwei Zhong, Xiangyu Kong, Yu Qiao,  Yizhou Wang},
  booktitle={The Thirty-Fourth AAAI Conference on Artificial Intelligence},
  year={2020}
}
```
