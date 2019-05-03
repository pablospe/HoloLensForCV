#!/usr/bin/env python2

# %% init

import os
import cv2
import pylab as plt

import re

import numpy as np
from open3d import Image,                   \
                   PointCloud,              \
                   Vector3dVector,          \
                   draw_geometries,         \
                   estimate_normals,        \
                   read_point_cloud,        \
                   write_point_cloud,       \
                   KDTreeSearchParamKNN,    \
                   KDTreeSearchParamHybrid, \
                   orient_normals_towards_camera_location

import open3d

from read_model import read_model, qvec2rotmat
from pcloud_compute import *

#import pickle

def get_sensor_poses(args, cam, time_stamp):
    # Input folder
    folder = args.workspace_path
    cam_folder = os.path.join(folder, cam)
    assert(os.path.exists(cam_folder))

    # From frame to world coordinate system
    sensor_poses = read_sensor_poses(os.path.join(folder, cam + ".csv"),
                                     identity_camera_to_image=True)        

    world2cam = sensor_poses[time_stamp]    
    
    return world2cam


# %%
def get_points2(img, us, vs, cam2world, depth_range):
    distance_img = pgm2distance(img, encoded=False)
    empty = np.zeros((1024,1024))
    depth_img = (empty).astype(np.float32)
    

    if cam2world is not None:
        R = cam2world[:3, :3]
        t = cam2world[:3, 3]
    else:
        R, t = np.eye(3), np.zeros(3)

    # from '[-1,1]x[-1,1]' to '[0,xdim]x[0,ydim]'
    xdim = depth_img.shape[1]
    ydim = depth_img.shape[0]
    fx = xdim/2
    fy = ydim/2
    cx = fx
    cy = fy
    cam_intrinsics = np.identity(3)
    cam_intrinsics[0,0] = fx
    cam_intrinsics[1,1] = fy
    cam_intrinsics[0,2] = cx
    cam_intrinsics[1,2] = cy


    points = []
    for i in np.arange(distance_img.shape[0]):
        for j in np.arange(distance_img.shape[1]):
            x = us[i, j]
            y = vs[i, j]
            D = distance_img[i, j]
            z = - float(D) / np.sqrt(x*x + y*y + 1)
                        
            if np.isinf(x) or np.isinf(y) or D < depth_range[0] or D > depth_range[1]:
#                depth_img[i,j] = 0
                continue
            
            
#            print("arr", np.array([x, y, 1.])) 
            
            point2D = cam_intrinsics.dot(np.array([x, y, 1.]))
            ii = np.int32(point2D[1])
            jj = np.int32(point2D[0])
            if ii < 0 or jj < 0:
#                print "What!!!"
#                print("original", np.array([x, y, 1.]))   # (u,v) bigger than 1 or -1 ?? Why?!! 
#                print("point2D", point2D)
                continue
            depth_img[ii,jj] = - z

            
            # 3D point in camera coordinate system
            point = np.array([x, y, 1.]) * z
                        
            # Camera to World
            point = np.dot(R, point) + t

            points.append(point)    
            
#    # for debug
#    plt.subplot(1, 3, 1)
#    plt.title('img')
#    plt.imshow(img)
#    plt.subplot(1, 3, 2)
#    plt.title('distance_img')
#    plt.imshow(distance_img)
#    plt.subplot(1, 3, 3)
#    plt.title('depth_img')
#    plt.imshow(depth_img)
#    plt.show()
    
    return np.vstack(points), depth_img


def process_folder2(args, cam):
    # Input folder
    folder = args.workspace_path
    cam_folder = os.path.join(folder, cam)
    assert(os.path.exists(cam_folder))
    # Output folder
    output_folder = os.path.join(args.output_path, cam)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get camera projection info
    bin_path = os.path.join(args.workspace_path, "%s_camera_space_projection.bin" % cam)
    
    # From frame to world coordinate system
    sensor_poses = None
    if not args.ignore_sensor_poses:
        sensor_poses = read_sensor_poses(os.path.join(folder, cam + ".csv"), identity_camera_to_image=True)        

    # Get appropriate depth thresholds
    depth_range = LONG_THROW_RANGE if 'long' in cam else SHORT_THROW_RANGE

    # Get depth paths
    depth_paths = sorted(glob(os.path.join(cam_folder, "*pgm")))
    if args.max_num_frames == -1:
        args.max_num_frames = len(depth_paths)
    depth_paths = depth_paths[args.start_frame:(args.start_frame + args.max_num_frames)]    

    us = vs = None
    # Process paths
    merge_points = args.merge_points
    rewrite = args.rewrite
    use_cache = args.use_cache
    points_merged = []
    normals_merged = []
    
    volume = open3d.ScalableTSDFVolume(voxel_length = 4.0 / 512.0,
             sdf_trunc = 0.1, color_type = open3d.TSDFVolumeColorType.None)  
#             depth_sampling_stride = 16)
    
    for i_path, path in enumerate(depth_paths):           
        output_suffix = "_%s" % args.output_suffix if len(args.output_suffix) else ""
#        pcloud_output_path = os.path.join(output_folder, os.path.basename(path).replace(".pgm", "%s.obj" % output_suffix))
        
        pcloud_output_path = \
            os.path.join(output_folder,
                         os.path.basename(path).replace(
                                 ".pgm","%s.ply" % output_suffix))   
        print("File: " + pcloud_output_path)

        # if file exist
        output_file_exist = os.path.exists(pcloud_output_path)
        if output_file_exist and use_cache:
            print("Progress (file cache): %d/%d" % (i_path+1, len(depth_paths)))
            pcd = read_point_cloud(pcloud_output_path)
            points  = pcd.points
            normals = pcd.normals
        else:
            print("Progress: %d/%d" % (i_path+1, len(depth_paths)))
            img = cv2.imread(path, -1)
            if us is None or vs is None:
                us, vs = parse_projection_bin(bin_path, img.shape[1], img.shape[0])
            cam2world = get_cam2world(path, sensor_poses) if sensor_poses is not None else None
            points, depth_img = get_points2(img, us, vs, cam2world, depth_range)
        
            # get normals (orientes towards the camera)
            t = cam2world[:3, 3]
            cam_center = t       # (0,0,0) is the camera center
            
            pcd = PointCloud()
            pcd.points = Vector3dVector(points)
            
            # compute normal of the point cloud (per camera)
            estimate_normals(pcd,
                             search_param =
                             # KDTreeSearchParamKNN(knn = 30))
                             KDTreeSearchParamHybrid(radius = 0.2, max_nn = 30))
            
            # orient normals
            orient_normals_towards_camera_location(pcd, cam_center)
    
    
    
            # get normal as python array
            normals = np.asarray(pcd.normals)
            
            # from '[-1,1]x[-1,1]' to '[0,xdim]x[0,ydim]'
            w = depth_img.shape[1] # xdim
            h = depth_img.shape[0] # ydim
            fx = w/2
            fy = h/2
            cx = fx
            cy = fy

            cam_intrinsic = open3d.PinholeCameraIntrinsic()
            cam_intrinsic.set_intrinsics(w, h, fx, fy, cx, cy)
            # print cam_intrinsic.intrinsic_matrix
            
            
            # convert to Open3d Image
            depth = open3d.Image((depth_img).astype(np.float32))


            # create dummy variable
            # color_type = open3d.TSDFVolumeColorType.None
            # we aren't using color
            color = Image( (np.ones((w,h))).astype(np.uint8) )
            # color = depth
            rgbd = open3d.create_rgbd_image_from_color_and_depth(color, depth,
                                                                 depth_scale=1.0, 
                                                                 depth_trunc=4,
                                                                 convert_rgb_to_intensity = False);
            
#            rgbd = open3d.create_rgbd_image_from_color_and_depth(color, depth,
#                                                            depth_scale=1.0, 
#                                                            depth_trunc = depth_range[1],
#                                                            convert_rgb_to_intensity = True);
#            

            # This transformation is requiere before of the line:
            #
            #     depth_img[... , ...] = - z
            #
            # in get_points(...)
            #
            img2cam = np.array(
                        [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            
            cam2world_new = np.dot(cam2world, img2cam)
            
            # print("volume.integrate")
            volume.integrate(rgbd, cam_intrinsic, np.linalg.inv(cam2world_new) )
        
        if merge_points:
            points_merged.extend(points)
            normals_merged.extend(normals)
        
        if rewrite:
            write_point_cloud(pcloud_output_path, pcd)

        
    return points_merged, normals_merged, volume


# %%
class ArgsTest:
  workspace_path = ''
  output_path = ''
  ignore_sensor_poses = False
  start_frame = 0
  max_num_frames = -1
  output_suffix = ''
  merge_points = True
  use_cache = False
  rewrite = False

args = ArgsTest()
#args.workspace_path = "../../data/HoloLensRecording__2018_10_23__14_05_39/"
args.workspace_path = "../../data/HoloLens_recording_sample/"

args.output_path = args.workspace_path

args.start_frame = 0
args.max_num_frames = 5


# %%
cam = 'long_throw_depth'
points_merged, normals_merged, volume = process_folder2(args, cam)

pcd_merged = PointCloud()
pcd_merged.points  = Vector3dVector(points_merged)
pcd_merged.normals = Vector3dVector(normals_merged)

#pcd.paint_uniform_color([1, 0.706, 0])
#draw_geometries([pcd_merged])


#folder = args.workspace_path
#write_point_cloud(folder + "\\" + cam + ".ply", pcd)

#draw_geometries([pcd2])

#pcd_vol = volume.extract_point_cloud()
#pcd_vol.transform([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
#print pcd_vol
#draw_geometries([pcd_vol])


# %%
pcd_vol = volume.extract_point_cloud()
#pcd_vol.transform([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
#print pcd_vol
pcd_merged.paint_uniform_color([1, 0.706, 0])
draw_geometries([pcd_vol, pcd_merged])


# %%
draw_geometries([pcd_vol])


# %%
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()
print mesh
draw_geometries([mesh])

