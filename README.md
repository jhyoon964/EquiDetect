# Equirectangular Point Reconstruction for Domain Adaptive Multimodal 3D Object Detection in Adverse Weather Conditions

Jae Hyun Yoon, Jong Won Jung, Seok Bong Yoo*

(abstract) A multimodal fusion technique using LiDAR-camera has been developed for precise 3D object detection in autonomous driving and provides acceptable detection performance in ideal conditions with clear weather. However, the existing multimodal methods are still vulnerable to adverse weather conditions, such as snow, rain, and fog. These factors increase the point cloud sparsity due to occlusion and attenuation of the laser signal. A point cloud becomes sparser with increased distance, posing a challenge for object detection. To address these problems, we propose a point reconstruction network using equirectangular projection for multimodal 3D object detection. This network consists of distance-constrained denoising to remove adverse weather noise and an object-centric ray generator to generate distant object points flexibly. We propose a domain adaptation method that injects feature perturbations to improve detection performance by reducing the domain gap between different datasets. Furthermore, we propose a multimodal weather noise matching method for realistic data synthesis-based training to align the adverse weather noise between synthetic point clouds and images. The experimental results on adverse weather datasets confirm that the proposed approach outperforms the existing methods.

The paper will be presented at [AAAI2025](https://aaai.org/conference/aaai/aaai-25/)



**Install the environment**
```
conda create -n equidetect python=3.8
conda activate equidetect
conda install -y pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -y pytorch3d -c pytorch3d
pip install numpy==1.19.5 protobuf==3.19.4 scikit-image==0.19.2 waymo-open-dataset-tf-2-2-0 nuscenes-devkit==1.0.5 spconv-cu111 numba scipy pyyaml easydict fire tqdm shapely matplotlib opencv-python addict pyquaternion awscli open3d pandas future pybind11 tensorboardX tensorboard Cython
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu111.html
```

**Setup**
```
cd EquiDetect && python setup.py develop --user
cd pcdet/ops/dcn && python setup.py develop --user
```

## Training & Testing
```
# Train
bash scripts/dist_train.sh

# Test
bash scripts/dist_test.sh
