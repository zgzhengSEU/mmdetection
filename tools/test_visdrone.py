# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS

import datetime as dt
NOW_TIME = dt.datetime.now().strftime('%Y%m%d_%H%M%S')

# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        nargs='?',
        type=str,
        const='default',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--json-prefix',
        nargs='?',
        type=str,
        const='default',
        help='the prefix of the output json file without perform evaluation, '
        'which is useful when you want to format the result to a specific '
        'format and submit it to the test server')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--softnms', action='store_true')
    # parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        if args.work_dir == 'default':
            work_name = f'{osp.splitext(osp.basename(args.config))[0]}'
            if args.softnms:
                work_name = f'{work_name}_SoftNMS'
            if args.tta:
                work_name = f'{work_name}_TTA'
            cfg.work_dir = osp.join('./work_dirs', f'model_test/{work_name}/{NOW_TIME}')
        else:
            cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    print(f'work_dir: {cfg.work_dir}')
    
    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    # add `format_only` and `outfile_prefix` into cfg
    if args.json_prefix is not None:
        if args.json_prefix == 'default':
            args.json_prefix = f'{cfg.work_dir}/{osp.splitext(osp.basename(args.config))[0]}_result'
        print(f'json output: {args.json_prefix}.json')
        cfg_json = {
            'test_evaluator.format_only': False,
            'test_evaluator.outfile_prefix': args.json_prefix
        }
        cfg.merge_from_dict(cfg_json)

    if args.tta:
        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            if args.softnms:
                cfg.tta_model = dict(
                    type='DetTTAModel',
                    tta_cfg=dict(
                        nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.001), max_per_img=500))
                print('USE SoftNMS In TTA')
            else:
                cfg.tta_model = dict(
                    type='DetTTAModel',
                    tta_cfg=dict(
                        nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        else:
            if args.softnms:
                cfg.tta_model = dict(
                    type='DetTTAModel',
                    tta_cfg=dict(
                        nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.001), max_per_img=500))
                print('USE SoftNMS In TTA')
                
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        print('USE TTA')
        
    if args.softnms:
        if 'test_cfg' in cfg.model:
            cfg.model.test_cfg.rcnn = dict(
                                        score_thr=0.05, 
                                        nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.001), 
                                        max_per_img=500) 
            print('USE SoftNMS In test_cfg.rcnn')
            
    if 'visualizer' in cfg:
        if 'Wandb_init_kwargs' in cfg:
            Wandb_init_kwargs = cfg.Wandb_init_kwargs
            Wandb_init_kwargs['group'] = 'Test'
            old_name = Wandb_init_kwargs['name']
            if args.softnms:
                old_name = f'{old_name}_SoftNMS'
                Wandb_init_kwargs['name'] = old_name
            if args.tta:
                old_name = f'{old_name}_TTA'
                Wandb_init_kwargs['name'] = old_name
            cfg.visualizer.vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend', init_kwargs=Wandb_init_kwargs)]         

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out))

    # start testing
    runner.test()
    
    print('============================== VisDrone2019-DET Toolkit =================================================')
    from visdrone.json_to_txt import Json2Txt
    from visdrone_eval.evaluate_all_in_one import visdrone_evaluate
    gt_json = 'data/VisDrone/annotations/test.json'
    annotations_dir = 'data/VisDrone/annotations/test'
    det_json = f'{args.json_prefix}.bbox.json'
    det_annotations_dir = '.work_dirs/model_test/visdrone_det_txt'
    
    # json -> txt
    if not os.path.isdir(det_annotations_dir):
        os.mkdir(det_annotations_dir)
    tool = Json2Txt(gt_json, det_json, det_annotations_dir)
    tool.to_txt()
    # eval
    tool = visdrone_evaluate(annotations_dir, det_annotations_dir)
    tool.run_eval()
    # clean
    os.system(f'rm -rf {det_annotations_dir}')
    
if __name__ == '__main__':
    main()
    
    
