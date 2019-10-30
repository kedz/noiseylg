from ..types import register, PlumObject, HP, props
import torch


@register("dataio.pipeline.long_tensor")
class LongTensor(PlumObject):
    def __call__(self, item):
        if isinstance(item, (list, tuple)):
            return torch.LongTensor(item)
        else:
            return torch.LongTensor([item])
