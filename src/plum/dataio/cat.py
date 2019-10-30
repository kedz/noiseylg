from ..types import register, PlumObject, HP, props
import plum


@register("dataio.pipeline.cat")
class Cat(PlumObject):

    dim = HP(type=props.INTEGER)

    def __call__(self, item):
        return plum.cat(item, dim=self.dim, )
