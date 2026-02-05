import torch
import time
import builtins
import datetime
import os
import argparse
import torch.distributed as dist

import numpy as np

from threading import Thread
import time

import pynvml

class Monitor(Thread):
    def __init__(self, delay=6, gpu_id=0):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.avg_util = np.ones(20) * 100.0  # 2 min avg
        self.ptr = 0
        self.gpu_id = gpu_id 

        self.start()


    def run(self):
        while not self.stopped:
            pynvml.nvmlInit()
            handler = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            util = pynvml.nvmlDeviceGetUtilizationRates(handler).gpu
            self.avg_util[self.ptr] = util
            self.ptr = (self.ptr + 1) % 20
            time.sleep(self.delay)
    
    def get_avg_util(self):
        return np.mean(self.avg_util)

    def stop(self):
        self.stopped = True


def get_gpu_mem_info(gpu_id=7):
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free
    


def get_torch_memory_usage(gpu_id):
    reserved = round(torch.cuda.memory_reserved(gpu_id) / 1024 / 1024, 2)
    allocted = round(torch.cuda.memory_allocated(gpu_id) / 1024 / 1024, 2)
    return reserved, allocted

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        # force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'

    # we don't need to init distributed env, running holder on each gpu independently


    # print('| distributed init (rank {}): {}, gpu {}'.format(
    #     args.rank, args.dist_url, args.gpu), flush=True)
    # torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                                      world_size=args.world_size, rank=args.rank)
    # torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('AdaptFormer fine-tuning for action recognition for image classification', add_help=False)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser = parser.parse_args()
    
    init_distributed_mode(parser)
    gpu = parser.gpu
    print(gpu)
    # gpu = 1

    mintor = Monitor(gpu_id=gpu, delay=6)
    util = 100

    memory_usage = np.ones(20) * 0.0
    ptr = 0
    is_init = False
    
    while(True):
        try:
            total, used, free = get_gpu_mem_info(gpu)
            # _, used_last, _ = get_gpu_mem_info(7)
            if used > 4000 and util > 20:
                print(total, used, free)
                torch.cuda.empty_cache()
                time.sleep(100)
                util = mintor.get_avg_util() # for safety, check ratio after sleeping 5min
                print(f'After sleeping 10 min, GPU util in last 2 min: {util}')
            else:
                print('Detect Empty, Running...')
                # 优化V2：真正达到90%+ GPU利用率，显存约5GB
                # 策略：启用梯度计算 + 多个计算流 + 大矩阵运算

                # 创建多层网络（约200MB参数）
                model = torch.nn.Sequential(
                    torch.nn.Conv2d(256, 512, 3, padding=1),
                    torch.nn.BatchNorm2d(512),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(512, 512, 3, padding=1),
                    torch.nn.BatchNorm2d(512),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(512, 256, 3, padding=1),
                ).to(gpu)

                # 精确控制显存在5GB左右
                # x1: 64*256*256*256*4 = 1.07GB
                x1 = torch.randn(64, 256, 256, 256, requires_grad=True).to(gpu)
                # x2, x3: 6144*6144*4 = 144MB each
                x2 = torch.randn(6144, 6144, requires_grad=True).to(gpu)
                x3 = torch.randn(6144, 6144, requires_grad=True).to(gpu)
                # x4, x5: 额外的矩阵用于并行计算
                x4 = torch.randn(6144, 6144, requires_grad=True).to(gpu)
                x5 = torch.randn(6144, 6144, requires_grad=True).to(gpu)

                count = 0
                while(True):
                    # 启用梯度计算来大幅提升GPU利用率
                    model.train()

                    # 密集计算1：卷积网络前向+反向
                    y1 = model(x1)
                    loss1 = y1.sum()
                    loss1.backward(retain_graph=True)

                    # 密集计算2：多个大矩阵乘法
                    y2 = torch.matmul(x2, x3)
                    y3 = torch.matmul(x4, x5)

                    # 额外的密集运算
                    y4 = torch.matmul(y2.t(), y3)
                    loss2 = y4.sum()
                    loss2.backward()

                    # 清零梯度，准备下一轮
                    model.zero_grad()
                    x1.grad = None
                    x2.grad = None
                    x3.grad = None
                    x4.grad = None
                    x5.grad = None

                    count += 1

                    # 大幅减少检查频率，从5次增加到200次
                    if count == 200:

                        total, used, free = get_gpu_mem_info(gpu)

                        if not is_init:
                            is_init = True
                            memory_usage = np.ones(20) * used
                            avg_usage = used
                        else:
                            avg_usage = np.mean(memory_usage)
                            memory_usage[ptr] = used
                            ptr = (ptr + 1) % 20
                        
                        # _, used_last, _ = get_gpu_mem_info(7)
                        # reserved, allocted = get_torch_memory_usage(gpu)
                        # print(total, used, free, reserved, allocted)
                        if used > avg_usage * 1.2:
                            print(total, used, free)
                            print(f'Detect Program Running, End holding. used/avg={used:.2f}/{avg_usage:.2f}')
                            util = 100
                            # 清理所有tensor和模型
                            del model, x1, x2, x3, x4, x5, y1, y2, y3, y4, loss1, loss2
                            torch.cuda.empty_cache()
                            break
                        count = 0
                

        except Exception as e:
            print(e)

