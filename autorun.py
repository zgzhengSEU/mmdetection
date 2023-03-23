import os
import sys
import time
import socket
import argparse
import pynvml

def grap_single_gpu(interval, gpu_index, cmd):
    hostname = socket.gethostname()
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    need_gpu_memory = 75
    while meminfo.free/(1024**2) < need_gpu_memory*1024:
        GPUBUSY = f'{hostname} GPU {gpu_index} BUSY {str(meminfo.free/(1024**3))[0:3]} GB need {need_gpu_memory} GB {time.strftime("%H:%M:%S", time.localtime())}'
        sys.stdout.write('\r' + GPUBUSY)
        sys.stdout.flush()
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        time.sleep(interval)
    
    print('\n\n\nGPU'+str(gpu_index)+' FREE '+str(meminfo.free/(1024**3))+'GB' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) 
    # time.sleep(10)
    print(str(cmd))
    os.system(str(cmd))    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    pythonpath = '/home/gp.sc.cc.tohoku.ac.jp/duanct/miniconda3/envs/mmdet3/bin/python'
    tools = 'tools/train.py'
    mmdet3path = '/home/gp.sc.cc.tohoku.ac.jp/duanct/openmmlab/mmdet3/'
    config = 'myconfig/VisDrone-H100/cascade_rcnn_new/cascade-rcnn_r50_fpn_1x_rsb_PAFPN_CARAFE_Skip_Parallel_ontput.py'
    amp = '--amp'
    cmd = f'{pythonpath} {tools} {mmdet3path}{config} {amp}'
    os.system(str(f'cd {mmdet3path}'))
    os.system('pwd')
    print(cmd) 
    grap_single_gpu(1, 0, cmd)
    print('主程序执行完毕')
