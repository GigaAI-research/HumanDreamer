import os
import sys

python_paths = [
    os.path.abspath(__file__).split('gd_scripts')[0],
    '/data/disk1/xinze.chen/codes/giga/giga-inference/',
    '/data/disk1/xinze.chen/codes/giga/giga-models/',
]
for python_path in python_paths:
    sys.path.insert(0, python_path)
    if 'PYTHONPATH' in os.environ:
        os.environ['PYTHONPATH'] += ':{}'.format(python_path)
    else:
        os.environ['PYTHONPATH'] = python_path
paths = ['/opt/conda/bin']
for path in paths:
    if 'PATH' in os.environ:
        os.environ['PATH'] += ':{}'.format(path)
    else:
        os.environ['PATH'] = path
os.environ['TORCH_HOME'] = '/data/disk3/models/torch/'
os.environ['TRANSFORMERS_CACHE'] = '/data/disk3/models/huggingface/'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/data/disk3/models/huggingface/'
os.environ['XDG_CACHE_HOME'] = '/data/disk3/models/xdg/'
os.environ['HF_DATASETS_CACHE'] = '/data/disk4/datasets/public_datasets/huggingface/'
os.environ['GIGA_DATASETS_DIR'] = '/data/disk2/datasets/giga_datasets/'
os.environ['GIGA_MODELS_DIR'] = '/data/disk3/models/'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
