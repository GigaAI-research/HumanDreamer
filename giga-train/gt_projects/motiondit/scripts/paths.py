import os
import sys

python_paths = [
    os.path.abspath(__file__).split('scripts')[0],
    'XXX/HumanDreamer/giga-datasets/',
    'XXX/HumanDreamer/giga-models/',
    'XXX/HumanDreamer/giga-train/',
    'XXX/HumanDreamer/giga-inference/',
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

