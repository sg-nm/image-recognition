from omegaconf import OmegaConf
import os
import os.path as osp

def load_config(cfg_file):
    cfg = OmegaConf.load(cfg_file)
    if '_base_' in cfg:
        if isinstance(cfg._base_, str):
            base_cfg = OmegaConf.load(osp.join(osp.dirname(cfg_file), cfg._base_))
        else:
            base_cfg = OmegaConf.merge(OmegaConf.load(f) for f in cfg._base_)
        cfg = OmegaConf.merge(base_cfg, cfg)
    return cfg

def get_config(args):
    cfg = load_config(args.cfg)
    OmegaConf.set_struct(cfg, True)

    if args.opts is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.opts))
    if hasattr(args, 'batch_size') and args.batch_size:
        cfg.data.batch_size = args.batch_size

    if hasattr(args, 'seed') and args.seed:
        cfg.seed = args.seed

    OmegaConf.set_readonly(cfg, True)
    return cfg