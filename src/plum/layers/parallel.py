from ..types import register, PlumModule, SM


@register("layers.parallel")
class Parallel(PlumModule):
    # TODO rename FanOut?
    layers = SM()

    def forward(self, inputs):
        return [module(inputs) for module in self.layers]
