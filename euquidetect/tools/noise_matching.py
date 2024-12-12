import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
import struct
import os

def load_kitti_bin(file_path):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points

def save_kitti_bin(file_path, points):
    with open(file_path, 'wb') as f:
        for point in points:
            f.write(struct.pack('ffff', *point))

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, np.sqrt(x**2 + y**2))
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)
    return x, y, z

def project_to_image(points, P2, R0_rect, Tr_velo_to_cam):
    points = np.insert(points, 3, 1, axis=1).T
    cam = P2 @ R0_rect @ Tr_velo_to_cam @ points
    cam[:2] /= cam[2, :]
    u, v = cam[:2]
    return u, v, cam[2]

def weather_noise_matching(points, noise_points, u, v, z, img_path, IMG_W, IMG_H, r_min=1.0, r_max=80.0):
    r, theta, phi = cartesian_to_spherical(points[:, 0], points[:, 1], points[:, 2])
    noise_r, noise_theta, noise_phi = cartesian_to_spherical(noise_points[:, 0], noise_points[:, 1], noise_points[:, 2])
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    noise_mask = img > 1
    
    snow_coords = np.column_stack(np.where(noise_mask))
    
    contours, _ = cv2.findContours(noise_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    projected_points = np.vstack((u, v)).T
    knn = NearestNeighbors(n_neighbors=4)
    knn.fit(projected_points)
    
    selected_noise_points = []
    for snow_coord in snow_coords:
        distances, indices = knn.kneighbors([snow_coord])
        for idx in indices.flatten():
            if idx >= len(noise_points):
                continue
            r_adjusted = np.clip(noise_r[idx], r_min, r_max)
            delta_beta = np.argmin(np.abs(r_adjusted - r[idx]))
            if r_adjusted + delta_beta > 0:
                selected_noise_points.append(noise_points[idx])

    selected_noise_points = np.array(selected_noise_points)
    all_points = np.vstack((points, selected_noise_points))

    return all_points

def process_point_cloud(bin_file_path, noise_bin_file_path, img_path, calib_file_path, output_bin_file_path):
    with open(calib_file_path, 'r') as f:
        calib = f.readlines()

    P2 = np.array([float(x) for x in calib[2].strip().split(' ')[1:]]).reshape(3, 4)
    R0_rect = np.array([float(x) for x in calib[4].strip().split(' ')[1:]]).reshape(3, 3)
    R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0], axis=0)
    R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0, 1], axis=1)
    Tr_velo_to_cam = np.array([float(x) for x in calib[5].strip().split(' ')[1:]]).reshape(3, 4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)

    points = load_kitti_bin(bin_file_path)[:, 0:3]
    noise_points = load_kitti_bin(noise_bin_file_path)[:, 0:3]

    u, v, z = project_to_image(points, P2, R0_rect, Tr_velo_to_cam)
    
    IMG_H, IMG_W = cv2.imread(img_path).shape[:2]

    matched_points = weather_noise_matching(points, noise_points, u, v, z, img_path, IMG_W, IMG_H)
    
    save_kitti_bin(output_bin_file_path, matched_points)

bin_folder_path = 'path/to/velodyne_no_noise/'
noise_folder_path = 'path/to/velodyne_noise/'
img_folder_path = 'path/to/image_noise_1/'
calib_folder_path = 'path/to/calib/'
output_folder_path = 'path/to/velodyne_noise_match/'

for file_name in os.listdir(bin_folder_path):
    if file_name.endswith('.bin'):
        name = os.path.splitext(file_name)[0]
        bin_file_path = os.path.join(bin_folder_path, file_name)
        noise_bin_file_path = os.path.join(noise_folder_path, f'{name}_noise.bin')
        img_path = os.path.join(img_folder_path, f'{name}.png')
        calib_file_path = os.path.join(calib_folder_path, f'{name}.txt')
        output_bin_file_path = os.path.join(output_folder_path, f'{name}_matched.bin')
        
        process_point_cloud(bin_file_path, noise_bin_file_path, img_path, calib_file_path, output_bin_file_path)
