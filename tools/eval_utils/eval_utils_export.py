import pickle
import time

import numpy as np
import torch
import tqdm
from pcdet.models.model_utils import model_nms_utils
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

from pcdet.models.backbones_2d.map_to_bev.bev_convS import points_to_bevs_two

import onnx
import onnx_tensorrt.backend as backend
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnx
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])
def build_engine(onnx_file_path):
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        # 创建配置对象
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 设置工作空间大小为 1GB
        if builder.platform_has_fast_fp16:
            print("enable fp16!!!")
            config.set_flag(trt.BuilderFlag.FP16)
        # 解析 ONNX 模型
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # 构建和返回 CUDA 引擎
        return builder.build_engine(network, config)


# 分配内存
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # 将输入数据传输到 GPU
    #[cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
    
    # 执行推理
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    
    # 将预测结果从 GPU 传回 CPU
    [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
    stream.synchronize()

    return [out['host'] for out in outputs]
def eval_one_epoch_test(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    
    post_process_cfg = model.model_cfg.POST_PROCESSING
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    times={}
    count=0
    model = onnx.load("/data/xqm/click2box/Fast_det/fastDet.onnx")
    engine = backend.prepare(model, device='CUDA:0')
    for i, batch_dict in enumerate(dataloader):
        count+=1
        load_data_to_gpu(batch_dict)
        if getattr(args, 'infer_time', False):
            start_time = time.time()
        input=points_to_bevs_two(batch_size=1,points=batch_dict['points'],point_range=[0, -40, -3, 70.4, 40, 1],size=[1408,1600])
        output_data = engine.run(input.cpu().numpy().astype(np.float32))
        batch_cls_preds = torch.tensor(output_data[0]).cuda()
        batch_box_preds = torch.tensor(output_data[1]).cuda()
        src_box_preds = batch_box_preds[0]
        src_cls_preds =batch_cls_preds[0]
        cls_preds = src_cls_preds
        cls_preds = torch.sigmoid(cls_preds)
        cls_preds, label_preds = torch.max(cls_preds, dim=-1)
        label_preds = label_preds + 1 
        selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=src_box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )
        pred_dicts = []  
        if post_process_cfg.OUTPUT_RAW_SCORE:
            max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
            selected_scores = max_cls_preds[selected]
        final_scores = selected_scores
        final_labels = label_preds[selected]
        final_boxes = src_box_preds[selected]
        record_dict = {
            'pred_boxes': final_boxes,
            'pred_scores': final_scores,
            'pred_labels': final_labels
        }
        pred_dicts.append(record_dict)
        
       
        
        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'
       
            
      
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()
    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict

def eval_one_epoch_trt(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    
    post_process_cfg = model.model_cfg.POST_PROCESSING
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    times={}
    count=0
    model = onnx.load("/data/xqm/click2box/Fast_det/fastDet.onnx")
    engine = build_engine("/data/xqm/click2box/Fast_det/fastDet.onnx")
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    for i, batch_dict in enumerate(dataloader):
        count+=1
        load_data_to_gpu(batch_dict)
        if getattr(args, 'infer_time', False):
            start_time = time.time()
        input=points_to_bevs_two(batch_size=1,points=batch_dict['points'],point_range=[0, -40, -3, 70.4, 40, 1],size=[1408,1600])
        numpy_input = input.cpu().numpy()
        cuda.memcpy_htod_async(inputs[0]['device'], numpy_input, stream)
        output_data = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        batch_cls_preds = torch.tensor(output_data[0]).cuda().reshape(1,-1,3)
        batch_box_preds = torch.tensor(output_data[1]).cuda().reshape(1,-1,7)
        src_box_preds = batch_box_preds[0]
        src_cls_preds =batch_cls_preds[0]
        cls_preds = src_cls_preds
        cls_preds = torch.sigmoid(cls_preds)
        cls_preds, label_preds = torch.max(cls_preds, dim=-1)
        label_preds = label_preds + 1 
        selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=src_box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )
        pred_dicts = []  
        if post_process_cfg.OUTPUT_RAW_SCORE:
            max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
            selected_scores = max_cls_preds[selected]
        final_scores = selected_scores
        final_labels = label_preds[selected]
        final_boxes = src_box_preds[selected]
        record_dict = {
            'pred_boxes': final_boxes,
            'pred_scores': final_scores,
            'pred_labels': final_labels
        }
        pred_dicts.append(record_dict)
        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'
       
            
      
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()
    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)


    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    times={}
    count=0
    for i, batch_dict in enumerate(dataloader):
        count+=1
        load_data_to_gpu(batch_dict)
        
        input_names = [ "input" ]
        output_names = [ "output"]
        xx=torch.rand([1,2,1600,1408]).cuda()
        with torch.no_grad():
            
            torch.onnx.export(model, xx, "/data/xqm/click2box/Fast_det/fastDet.onnx", verbose=True, input_names=input_names, output_names=output_names, opset_version=11)
       
        if cfg.LOCAL_RANK == 0:
            progress_bar.update()
        break
    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    
    if cfg.LOCAL_RANK != 0:
        return {}

    

    
    logger.info('****************Export done.*****************')
    


if __name__ == '__main__':
    pass

