from ..types import register, PlumObject, HP, props
import torch


@register("dataio.pipeline.pad_list")
class PadList(PlumObject):

    pad = HP()
    start = HP(default=True, type=props.BOOLEAN)
    end = HP(default=True, type=props.BOOLEAN)

    def __call__(self, item):
        if self.start:
            item = [self.pad] + item 
        if self.end:
            item = item + [self.pad]
        return item
