from ..types import register, PlumObject, HP, props
import torch


@register("dataio.pipeline.batch_ndtensor")
class BatchNDTensor(PlumObject):
    
    batch_dim = HP(default=0, type=props.INTEGER)

    def __call__(self, batch):
        batch = [item.unsqueeze(self.batch_dim) for item in batch] 
        batch = torch.cat(batch, dim=self.batch_dim)
        return batch
