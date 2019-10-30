from ..types import register, PlumObject, HP, props
import torch

@register("dataio.pipeline.one_hot")
class OneHot(PlumObject):

    index_field = HP()
    length_field = HP()
    pad_start = HP(default=False, type=props.BOOLEAN)
    pad_end = HP(default=False, type=props.BOOLEAN)

    def __call__(self, item):
        print(len(item['lt']))
        indices = item[self.index_field]
        max_len = item[self.length_field]
        start_offset = 1 if self.pad_start else 0

        if self.pad_start:
            max_len += 1

        if self.pad_end:
            max_len += 1

        out = [0] * max_len

        for i in indices:
            out[i+start_offset] = 1
        print(max_len,len(out))
        return torch.LongTensor(out)
