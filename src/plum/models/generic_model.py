from ..types import register, PlumModule, HP, SM, LazyDict
from .plum_model import PlumModel


def _curry_forward(net, fields, batch):
    def forward():
        if fields is not None:
            args = [batch[field] for field in fields]
            return net(*args)
        else:
            return net(batch)
    return forward

@register("plum.models.generic_model")
class GenericModel(PlumModel):
    
    inputs = HP(default=None)
    networks = SM()

    def forward(self, batch):
        output = LazyDict()
        inputs = self.inputs if self.inputs is not None else {}
        for name, net in self.networks.items():
            fields = inputs.get(name, None)
            output.lazy_set(name, _curry_forward(net, fields, batch))
        return output
