import yaml
import argparse
from easydict import EasyDict

from utils.transforms import *

if __name__ == "__main__":

    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Adding optional argument
    parser.add_argument("-o", "--config", help = "Config file")
    
    # Read arguments from command line
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    normalizer = stain_normalizer(config.ref_image)

    image_ids = get_file_list(config.input_path)

    for key in image_ids.keys():
        for idx in range(image_ids[key]):

            img = load_data(config.input_path, key, idx)

            img = rgb2ycbcr(img)

