from accelerate.utils import release_memory, wait_for_everyone

from motiondit import DITTrainer, DITTester
from giga_train import load_config
from giga_train.utils import parse_args


def train_net(config):
    trainer = DITTrainer.load(config)
    trainer.save_config(config)
    trainer.print(load_config(config))
    config = load_config(config)
    if config.get('resume', False):
        trainer.resume()
    trainer.train()

def test_net(config):
    tester = DITTester.load(config)
    tester.print(load_config(config))
    tester.test()


def train_test_net(config, stages):
    stages = stages.split(',')
    for stage in stages:
        if stage == 'train':
            train_net(config)
        elif stage == 'test':
            test_net(config)
        else:
            assert False
        release_memory()
        wait_for_everyone()

def main():
    args = parse_args()
    print('trian test frame cond')
    train_test_net(args.config, args.stages)


if __name__ == '__main__':
    main()
