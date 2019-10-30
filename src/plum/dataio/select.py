from ..types import register, PlumObject, HP, props
import torch


@register("dataio.pipeline.select")
class Select(PlumObject):

    fields = HP()
    type = HP(required=False)

    def __pluminit__(self, type):
        if type == "LongTensor":
            self._type_convert = torch.LongTensor
        elif type == "FloatTensor":
            self._type_convert = torch.FloatTensor
        elif type == "ByteTensor": 
            self._type_convert = torch.ByteTensor
        else:
            self._type_convert = None

    def __call__(self, item):

        data = [item[field] for field in self.fields]
        if self._type_convert is not None:
            data = self._type_convert(data)

        return data

    def __repr__(self):
        return "Select({}, type={})".format(str(self.fields), self.type)
