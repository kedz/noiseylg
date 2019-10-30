from ..types import register, PlumObject, HP, props, Variable
import plum


@register("dataio.pipeline.pad_dim_to_max")
class PadDimToMax(PlumObject):

    pad_value = HP()
    pad_dim = HP()

    def __call__(self, batch):

        sizes = [item.size(self.pad_dim) for item in batch]
        max_size = max(sizes)
        diffs = [max_size - sz for sz in sizes]

        for i, diff in enumerate(diffs):
            if diff == 0:
                continue
            
            dims = list(batch[i].size())
            dims[self.pad_dim] = diff
            
            pad = batch[i].new(*dims).fill_(self.pad_value)
            batch[i] = plum.cat([batch[i], pad], dim=self.pad_dim, 
                                ignore_length=True)

        return batch
