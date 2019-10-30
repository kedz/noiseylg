from ..types import register, PlumModule

@register("layers.identity")
class Identity(PlumModule):
    def forward(self, inputs):
        return inputs
