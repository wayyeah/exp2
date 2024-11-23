from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
import copy
import numpy as np
import pickle
from numba import cuda
import warnings
from numba.core.errors import NumbaPerformanceWarning
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils
from tqdm import tqdm
import torch
import os
# Suppress only NumbaPerformanceWarning
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)
def read_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

pred=read_pkl('/mnt/32THHD/yw/exp2/output/waymo_models/fade/waymo_480/eval/eval_all_default/default/epoch_461/val/result.pkl')

def box_cut(box, cloud_in, scale=1.0):
    """
    input:
        box: array, shape=(7,)  (x, y, z, l, w, h, yaw)
        cloud: array, shape(N,M), (x, y, z, intensity, ...)
        scale: float, factor to enlarge the box size
    output:
        pts_in: array, points in box
        pts_out: array, points outside box
    """

        # 确保数据在 GPU 上，如果有可用 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 假设 cloud_in 是一个 NumPy 数组，首先将其转换为 PyTorch 张量
    cloud_in = torch.tensor(cloud_in, dtype=torch.float32, device=device)
    box=torch.tensor(box,dtype=torch.float32,device=device)
    # 创建一个形状为 (N, 4) 的全零张量
    cloud = torch.zeros((cloud_in.shape[0], 4), dtype=torch.float32, device=device)

    # 将 cloud_in 的前三列复制到 cloud 的前三列
    cloud[:, 0:3] = cloud_in[:, 0:3]
    cloud[:, 3] = 1  # 设置 cloud 的第四列为 1

    # 获取 box 参数
    x, y, z, l, w, h, yaw = box[0], box[1], box[2], box[3], box[4], box[5], box[6]

    # 创建变换矩阵
    trans_mat = torch.eye(4, dtype=torch.float32, device=device)
    trans_mat[0, 0] = torch.cos(yaw)
    trans_mat[0, 1] = -torch.sin(yaw)
    trans_mat[0, 3] = x
    trans_mat[1, 0] = torch.sin(yaw)
    trans_mat[1, 1] = torch.cos(yaw)
    trans_mat[1, 3] = y
    trans_mat[2, 3] = z

    # 计算变换矩阵的逆
    trans_mat_i = torch.inverse(trans_mat)

    # 进行变换操作
    cloud = torch.matmul(cloud, trans_mat_i.T)

    # 使用 PyTorch 操作进行掩码操作
    mask_l = (cloud[:, 0] < l * scale / 2) & (cloud[:, 0] > -l * scale / 2)
    mask_w = (cloud[:, 1] < w * scale / 2) & (cloud[:, 1] > -w * scale / 2)
    mask_h = (cloud[:, 2] < h * scale / 2) & (cloud[:, 2] > -h * scale / 2)

    # 合并掩码
    mask = mask_l & mask_w & mask_h
    mask_not = ~mask

    

    return mask
for i in tqdm(range(len(pred))):
    
    #print(pred[i].keys())
    folder=pred[i]['frame_id'][:len(pred[i]['frame_id'])-4]
    frame='0'+pred[i]['frame_id'][-3:]
    lidar_file='/mnt/32THHD/yw/exp2/data/waymo/waymo_processed_data_v0_5_0/'+folder+'/'+frame+'.npy'
    points_all=np.load(lidar_file)
    pred_boxes=pred[i]['boxes_lidar']
    pred_boxes[:,3]=pred_boxes[:,3]+1
    pred_boxes[:,4]=pred_boxes[:,4]+1
    pred_boxes[:,5]=pred_boxes[:,5]+1
    mask_all=torch.zeros(len(points_all)).cuda()
    for j in range(pred_boxes.shape[0]):
        mask=box_cut(pred_boxes[j],points_all[:,:3])
        mask_all=mask_all+mask
    point_save=points_all[mask_all.cpu().numpy()>0]
    new_folder='/mnt/32THHD/yw/exp2/data/waymo/waymo_processed_data_v0_5_0_fade/'+folder
    if not os.path.exists(new_folder):
      
        os.mkdir(new_folder)
    np.save('/mnt/32THHD/yw/exp2/data/waymo/waymo_processed_data_v0_5_0_fade/'+folder+'/'+frame+'.npy',point_save)
    # exit()
    # lidar_file ='/mnt/16THDD/yw/Fast_det/data/kitti/training/velodyne/'+pred[i]['frame_id']+'.bin'
    # points_all=np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
    
    # pred_boxes=pred[i]['boxes_lidar']
    # #np.save('old_points.npy',points_all)
    # #np.save("pred_boxes.npy",pred_boxes)
    # pred_boxes[:,3]=pred_boxes[:,3]+1
    # pred_boxes[:,4]=pred_boxes[:,4]+1
    # pred_boxes[:,5]=pred_boxes[:,5]+1
   
    # boxes3d, is_numpy = common_utils.check_numpy_to_torch(pred_boxes)
    # points, is_numpy = common_utils.check_numpy_to_torch(points_all)
    # point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3], boxes3d)
    # points = points[point_masks.sum(dim=0) == 1]
    # save_path='/mnt/16THDD/yw/Fast_det/data/kitti/training/velodyne_fade/'+pred[i]['frame_id']+'.bin'
    # with open(save_path, 'w') as f:
    #     points.numpy().tofile(f)
    
 