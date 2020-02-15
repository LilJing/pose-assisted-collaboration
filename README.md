# Pose-Assisted-Multi-Camera-Collaboration
This package contains code to the paper "Jing Li*, Jing Xu*, Fangwei Zhong*, Xiangyu Kong, Yu Qiao, Yizhou Wang. Pose-Assisted Multi-Camera Collaboration for Active Object Tracking. AAAI 2020." [[PDF]](https://arxiv.org/abs/2001.05161).

## Multi-Camera Active Tracking
![Task](https://github.com/LilJing/pose-assisted-collaboration/blob/master/images/task.jpg)

## Setup
The code was built on Ubuntu 16.04.6 LTS with Python 3.5.6, PyTorch 1.0.1, and CUDA 8.0.61.
### Environment setup
Download the Virtual Environment -- UnrealCV in [gym-unrealcv](https://github.com/LilJing/gym_unrealcv):
```
git clone https://github.com/LilJing/gym_unrealcv
cd gym_unrealcv
pip install -e .
```

## Training Reproduction

If you want to train the vision model, clone the repo:
```
git clone https://github.com/LilJing/pose-assisted-collaboration.git 
cd pose-assisted-collaboration
```
and then use following command:
```
python main.py --rescale --shared-optimizer --env UnrealMCRoomLarge-DiscreteColorGoal-v5
```
If you want to train the pose model, clone the branch:
```
git clone -b pose https://github.com/LilJing/pose-assisted-collaboration.git 
cd pose-assisted-collaboration/pose
```
and then use following command:
```
python main.py --env PoseEnv-v1 --shared-optimizer --render

```
All models will be saved in corresponding logs dir.

### Trained Models
Our trained best models are in ./models;

The trained vision-based controller model is Vision-model-best.dat and the pose-based controller model is Pose-model-best.dat.

## Evaluation Reproduction

There are two environments for evaluation. You can choose environment UnrealGarden-DiscreteColorGoal-v1 (for _Garden_ environment) or UnrealUrbanTreeOBST-DiscreteColorGoal-v1 (for _Urban City_ environment).

### Run Our Method for Evaluation
Run the pose-assisted multi-camera collaboration method on _Garden_:
```
python evaluate.py --rescale --load-vision-model  ./models/Vision-model-best.dat --load-pose-model ./models/Pose-model-best.dat  --env UnrealGarden-DiscreteColorGoal-v1 --num-episodes 100 --test-type modelgate --render
```
Run the pose-assisted multi-camera collaboration method on _Urban City_:
```
python evaluate.py --rescale --load-vision-model  ./models/Vision-model-best.dat --load-pose-model ./models/Pose-model-best.dat  --env UnrealUrbanTreeOBST-DiscreteColorGoal-v1 --num-episodes 100 --test-type modelgate --render
```

## More Visualization

To see complete videos, please refer to [YouTube](https://www.youtube.com/watch?v=8Ha7HGkRv6k&feature=youtu.be).
