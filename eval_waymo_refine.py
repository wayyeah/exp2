
import copy
import numpy as np
import pickle
from numba import cuda
import warnings
from numba.core.errors import NumbaPerformanceWarning
from tqdm import tqdm
from pcdet.ops.iou3d_nms import iou3d_nms_utils
import torch
class_names=['Vehicle', 'Pedestrian', 'Cyclist']
class_names=['Vehicle']
# Suppress only NumbaPerformanceWarning
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)
def read_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

gt_path='/mnt/32THHD/yw/Fast_det/data/waymo/waymo_processed_data_v0_5_0_infos_val.pkl'
result_path='/mnt/16THDD/yw/Fast_det/result.pkl'
try:
    gt_pkl=read_pkl(gt_path)
except:
    gt_path=None
try:
    det_pkl=read_pkl(result_path)
except:
    result_path=None
if gt_path is None:
    print("输入gt路径")
    gt_path=input()
if result_path is None:
    print("输入result路径")
    result_path=input()



gt_pkl=read_pkl(gt_path)
det_pkl=read_pkl(result_path)
eval_det_annos = copy.deepcopy(det_pkl)
eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_pkl]
# eval_det_annos=eval_det_annos[:500]
# eval_gt_annos=eval_gt_annos[:500]
from pcdet.datasets.waymo.waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
eval = OpenPCDetWaymoDetectionMetricsEstimator()


for i in tqdm(range(len(eval_det_annos))):
    #print("pred shape",eval_det_annos[i]['boxes_3d'].shape)
    #print("gt shape",eval_gt_annos[i]['boxes_3d'].shape)
    # for index in range(len(eval_det_annos[i]['boxes_lidar'])):
    #     eval_det_annos[i]['boxes_lidar'][index][3]=eval_det_annos[i]['boxes_lidar'][index][3]
    #     continue
        #eval_det_annos[i]['boxes_lidar'][index][3]=eval_det_annos[i]['boxes_lidar'][index][3]-0.05
        #eval_det_annos[i]['boxes_lidar'][index][4]=eval_det_annos[i]['boxes_lidar'][index][4]+0.05
    ious=iou3d_nms_utils.boxes_iou3d_gpu(torch.tensor(eval_det_annos[i]['boxes_lidar'],dtype=torch.float32).cuda(),torch.tensor(eval_gt_annos[i]['gt_boxes_lidar'][:,:7],dtype=torch.float32).cuda())
    for index,iou in enumerate(ious):
        try:
            max_iou=torch.max(iou)
            max_iou_index=torch.argmax(iou)
        except:
            continue
        #print("max iou{}, max iou index{}".format(max_iou,max_iou_index))
        if max_iou>0.5:
            eval_det_annos[i]['boxes_lidar'][index][6]=eval_gt_annos[i]['gt_boxes_lidar'][max_iou_index][6]
            
    
    #     #print(eval_gt_annos[i]['name'])
    #     #print(eval_det_annos[i]['frame_id'])
    #     #print(i)
    #     #print(torch.tensor(eval_gt_annos[i]['boxes_3d'],dtype=torch.float32).cuda())
    #     #input()

ap_dict = eval.waymo_evaluation(
    eval_det_annos, eval_gt_annos, class_name=class_names,
    distance_thresh=1000, fake_gt_infos=False
)
ap_result_str = '\n'
for key in ap_dict:
    ap_dict[key] = ap_dict[key][0]
    ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

print(ap_result_str)