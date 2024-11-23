from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
import copy
import numpy as np
import pickle
from numba import cuda
import warnings
from numba.core.errors import NumbaPerformanceWarning
class_names=['Vehicle', 'Pedestrian', 'Cyclist']
class_names=['Vehicle']
# Suppress only NumbaPerformanceWarning
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)
def read_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

gt_path='/mnt/32THHD/yw/Fast_det/data/waymo/waymo_processed_data_v0_5_0_infos_val.pkl'
result_path=None
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
#eval_det_annos=eval_det_annos[:10000]
#eval_gt_annos=eval_gt_annos[:10000]
from pcdet.datasets.waymo.waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
eval = OpenPCDetWaymoDetectionMetricsEstimator()

ap_dict = eval.waymo_evaluation(
    eval_det_annos, eval_gt_annos, class_name=class_names,
    distance_thresh=1000, fake_gt_infos=False
)
ap_result_str = '\n'
for key in ap_dict:
    ap_dict[key] = ap_dict[key][0]
    ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

print(ap_result_str)