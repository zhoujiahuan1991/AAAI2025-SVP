import argparse
import utils
from pathlib import Path
import logging
import os
import sys
import os.path as op

def print_args_with_newlines(args):
    # 获取所有属性值，转换为字符串，并用换行符连接
    output_string = '\n'.join(attr+":"+str(getattr(args, attr)) for attr in vars(args))
    print(output_string)

def log_args(args):
    if args.output_dir and utils.is_main_process():
        with (Path(args.output_dir) / "log.txt").open("a") as f:
            f.write("arguments:"+'\n')
            f.write('\n'.join(f"{key}: {value}" for key, value in vars(args).items())+'\n')


def setup_logger(name, save_dir, if_train, distributed_rank=0, if_no_logging=False):
    """
    设置并初始化日志记录器。

    参数:
    - name: 日志记录器的名称。
    - save_dir: 保存日志文件的目录路径。
    - if_train: 布尔值，指示是否是训练模式。
    - distributed_rank: 分布式训练中的排名，默认为0，表示主进程。

    返回:
    - 配置好的日志记录器实例。
    """
    logger = logging.getLogger(name)  # 获取或创建指定名称的日志记录器

    # if if_no_logging == False:
    logger.setLevel(logging.DEBUG)  # 设置日志记录级别为DEBUG
    # elif if_no_logging == True:
    #     logger.setLevel(logging.CRITICAL)

    # 如果是分布式训练且不是主进程，则不记录日志
    if distributed_rank > 0:
        return logger

    # 配置控制台日志处理器
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)  # 控制台日志输出级别为DEBUG
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")  # 设置日志格式
    ch.setFormatter(formatter)
    logger.addHandler(ch)  # 添加控制台日志处理器

    # 检查日志保存目录是否存在，不存在则创建
    if if_no_logging == False:
        if not os.path.exists(save_dir):
            print(f"{save_dir} is not exists, create given directory")
            os.makedirs(save_dir)

        # 根据训练或测试模式，配置并添加文件日志处理器
        if if_train:
            fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='w')  # 训练模式，创建文件日志处理器
        else:
            fh = logging.FileHandler(os.path.join(save_dir, "test_log.txt"), mode='a')  # 测试模式，追加模式打开文件日志处理器

        fh.setLevel(logging.DEBUG)  # 文件日志输出级别为DEBUG
        fh.setFormatter(formatter)  # 设置文件日志格式
        logger.addHandler(fh)  # 添加文件日志处理器

    return logger