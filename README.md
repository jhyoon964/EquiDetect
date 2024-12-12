# Equirectangular Point Reconstruction for Domain Adaptive Multimodal 3D Object Detection in Adverse Weather Conditions

(abstract) A multimodal fusion technique using LiDAR-camera has been developed for precise 3D object detection in autonomous driving and provides acceptable detection performance in ideal conditions with clear weather. However, the existing multimodal methods are still vulnerable to adverse weather conditions, such as snow, rain, and fog. These factors increase the point cloud sparsity due to occlusion and attenuation of the laser signal. A point cloud becomes sparser with increased distance, posing a challenge for object detection. To address these problems, we propose a point reconstruction network using equirectangular projection for multimodal 3D object detection. This network consists of distance-constrained denoising to remove adverse weather noise and an object-centric ray generator to generate distant object points flexibly. We propose a domain adaptation method that injects feature perturbations to improve detection performance by reducing the domain gap between different datasets. Furthermore, we propose a multimodal weather noise matching method for realistic data synthesis-based training to align the adverse weather noise between synthetic point clouds and images. The experimental results on adverse weather datasets confirm that the proposed approach outperforms the existing methods.


## Training & Testing
```
# Train
bash scripts/dist_train.sh

# Test
bash scripts/dist_test.sh
