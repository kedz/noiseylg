from ..types import register, PlumModule, P, HP, props
import plum


@register("layers.concat")
class Concat(PlumModule):

    dim = HP(type=props.INTEGER)

    def forward(self, inputs):
        return plum.cat(inputs, dim=self.dim)
