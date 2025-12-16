import paths  # noqa F401
from giga_train import Config, Launcher
import torch
from train_test_net_frame_cond import train_test_net
import argparse
def main():
    config_path = 'configs.f64_dit_t2m_score_50k.config'
    # stages = ['train']
    stages = ['test']
    
    parser = argparse.ArgumentParser(description='接收配置文件路径')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--stages', type=str, help='训练阶段')
    args = parser.parse_args()
    if args.config:
        config_path = args.config
    if args.stages:
        stages = args.stages.split(',')
    print(f'{stages}: {config_path}')

    stages = ','.join(stages)
    config = Config.load(config_path)
    gpu_ids = config.launch.gpu_ids
    if len(gpu_ids) == 1:
        torch.cuda.set_device(gpu_ids[0])
        train_test_net(config_path, stages)
    else:
        launcher = Launcher(**config.launch)
        launcher.launch('train_test_net_frame_cond.py --config {} --stages {}'.format(config_path, stages))

if __name__ == '__main__':
    main()
