import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from pathlib import Path 

from model import build_pointernet
from dataset import Dataset
from config import get_config


config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(config):
    dataset = Dataset(config["training_path"])
    return None
