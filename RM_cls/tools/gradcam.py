import sys

sys.path.append('.')

from rmclas.config import get_cfg
from rmclas.engine import default_argument_parser, default_setup
from rmclas.utils.gradcam import gradcam_lanuch


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)
    gradcam_lanuch(
        cfg.GC.IMAGE_DIR,
        cfg.GC.DICT_DIR,
        cfg.GC.OUTPUT_DIR,
        cfg.GC.IMAGE_SIZE,
        cfg.GC.TARGET_LAYER,
    )


#'image_dir, dict_dir, model_class, output_dir, image_dize'