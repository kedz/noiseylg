from ..types import register, PlumObject, HP, props


@register("dataio.pipeline.len")
class Len(PlumObject):

    def __call__(self, item):
        return len(item)
