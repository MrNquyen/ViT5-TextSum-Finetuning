from utils.flags import Flags
from utils.configs import Config
from utils.utils import load_yml
from utils.trainer_vit5 import Trainer

if __name__=="__main__":
    # -- Get Parser
    flag = Flags()
    args = flag.get_parser()

    # -- Get device
    # print(args)
    config = args.config
    device = args.device

    # -- Get config
    config = load_yml(args.config)
    config_container = Config(config)

    # -- Trainer
    trainer = Trainer(config, args)
    if args.run_type=="train":
        trainer.train()
    elif args.run_type=="inference":
        trainer.inference(mode="test", save_dir="/datastore/npl/ViInfographicCaps/workspace/baseline/LSTMR/save/results")