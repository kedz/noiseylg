from .parser import PlumParser
from . import dataio
from . import layers
from . import trainer
from .vocab import Vocab
from . import loss_functions
from . import metrics
from . import optimizers
from . import seq2seq
from . import models
from . import initializers
from . import checkpoints
from . import tasks
from .tensor_ops import *
from . import loggers


def load(path):
    import torch
    import json
    data = torch.load(path, map_location="cpu")
    obj, _ = parser.PlumParser()._build_config(json.loads(data["plum_data"]))
    if data["state_dict"] is not None:
        obj.load_state_dict(data["state_dict"])
    return obj
