from ..types import register, PlumObject
import torch


@register("dataio.pipeline.batch_flat")
class BatchFlat(PlumObject):
    def __call__(self, batch):
        batch = [item.view(-1) for item in batch] 
        batch = torch.cat(batch)
        return batch
