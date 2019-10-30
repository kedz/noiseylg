from ..types import register, PlumModule, SM

@register("layers.sequential")
class Sequential(PlumModule):
    layers = SM()

    def forward(self, inputs):
        res = inputs
        for module in self.layers:
            res = module(res)
        return res
