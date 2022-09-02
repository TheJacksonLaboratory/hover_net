import importlib
import random

import cv2
import torch
import numpy as np

from dataset import get_dataset
from models import Synthesizer


class Config(object):
    """Configuration file."""

    def __init__(self):
        self.seed = 10

        self.logging = True

        # turn on debug flag to trace some parallel processing problems more easily
        self.debug = False

        model_name = "hovernet"
        model_mode = "compressed_rec" # choose either `original` or `fast`
        # model_mode = "fast" # choose either `original` or `fast`

        if model_mode not in ["original", "fast", "compressed_rec"]:
            raise Exception("Must use either `original`, `fast`, or `compressed_rec` as model mode")

        nr_type = 5 # number of nuclear types (including background)

        # whether to predict the nuclear type, availability depending on dataset!
        self.type_classification = True

        # shape information - 
        # below config is for original mode. 
        # If original model mode is used, use [270,270] and [80,80] for act_shape and out_shape respectively
        # If fast model mode is used, use [256,256] and [164,164] for act_shape and out_shape respectively
        self.aug_shape = [540, 540] # patch shape used during augmentation (larger patch may have less border artefacts)
        self.stride = [164, 164] # stride between patch windows extracted (for zarr based loader)
        act_shape = [256, 256] # patch shape used as input to network - central crop performed after augmentation
        out_shape = [164, 164] # patch shape at output of network

        if model_mode == "original":
            if act_shape != [270,270] or out_shape != [80,80]:
                raise Exception("If using `original` mode, input shape must be [270,270] and output shape must be [80,80]")
        if model_mode == "fast":
            if act_shape != [256,256] or out_shape != [164,164]:
                raise Exception("If using `fast` mode, input shape must be [256,256] and output shape must be [164,164]")

        self.dataset_name = "monusac" # extracts dataset info from dataset.py
        # self.log_dir = "/mnt/logs" # where checkpoints will be saved
        self.log_dir = "/mnt/logs" # where checkpoints will be saved

        rec_model = None
        # rec_model_path = None
        rec_model_path = "/mnt/models/best_ver0.5.5_32203.pth"

        if rec_model_path is not None:
            saved_state_dict = torch.load(rec_model_path, map_location='cpu')
            rec_model = Synthesizer(**saved_state_dict["args"])
            rec_model.load_state_dict(saved_state_dict["decoder"])
            rec_model = torch.nn.DataParallel(rec_model)
            if torch.cuda.is_available():
                rec_model.cuda()
            rec_model.eval()

        # paths to training and validation patches
        self.train_dir_list = [
            "/mnt/data/train_list.txt"
        ]
        self.valid_dir_list = [
            "/mnt/data/valid_list.txt"
        ]
        
        self.src_fmt = ".zarr"
        self.split_train=168
        self.split_val=41
        self.labels_group = "labels/0/0"
        self.data_group = "compressed/0/0"
        self.compression_level = 3
        self.compressed_input = True
        # self.data_group = "0/0"
        # self.compression_level = 0
        # self.compressed_input = False

        self.shape_info = {
            "train": {"input_shape": act_shape, "mask_shape": out_shape,},
            "valid": {"input_shape": act_shape, "mask_shape": out_shape,},
        }

        # * parsing config to the running state and set up associated variables
        self.dataset = get_dataset(self.dataset_name)

        module = importlib.import_module(
            "models.%s.opt" % model_name
        )
        self.model_config = module.get_config(nr_type, model_mode, rec_model=rec_model)
