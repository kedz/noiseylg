from ..types import register, PlumObject, HP, props, Variable
import torch


@register("dataio.pipeline.batch_sequence_ndtensor")
class BatchSequenceNDTensor(PlumObject):
    
    batch_dim = HP(default=0, type=props.INTEGER)
    sequence_dim = HP(type=props.INTEGER)
    pad_value = HP()
    pad_right = HP(default=True, type=props.BOOLEAN)

    def __call__(self, batch):

        ref_dims = list(batch[0].size())
        ref_dims.pop(self.sequence_dim)

        for item in batch[1:]:
            item_dims = list(item.size())
            item_dims.pop(self.sequence_dim)
            if item_dims != ref_dims:
                raise RuntimeError(
                    "Item dims must match everywhere except sequence " + \
                    "dim ({}). ".format(self.sequence_dim))

        lengths = [item.size(self.sequence_dim) for item in batch]
        max_length = max(lengths)
        pad_lengths = [max_length - l for l in lengths]

        for i, pad_length in enumerate(pad_lengths):
            if pad_length == 0:
                batch[i] = batch[i].unsqueeze(self.batch_dim)
                continue

            pad_dims = list(batch[i].size())
            pad_dims[self.sequence_dim] = pad_length
            pad_tensor = batch[i].new(*pad_dims).fill_(self.pad_value)

            if self.pad_right:
                batch[i] = torch.cat([batch[i], pad_tensor], 
                                     dim=self.sequence_dim)
            else:
                batch[i] = torch.cat([pad_tensor, batch[i]], 
                                     dim=self.sequence_dim)
            batch[i] = batch[i].unsqueeze(self.batch_dim)

        seq_dim = self.sequence_dim
        if self.batch_dim <= self.sequence_dim:
            seq_dim += 1
        tensor = torch.cat(batch, dim=self.batch_dim)
        
        return Variable(tensor, lengths=torch.LongTensor(lengths), 
                        length_dim=seq_dim, batch_dim=self.batch_dim,
                        pad_value=self.pad_value)
