from ..types import register, PlumObject, HP, props


@register("dataio.pipeline.avg_getters")
class AverageGetters(PlumObject):

    fields = HP()

    def apply_fields(self, item, fields):
        for field in fields:
            if hasattr(field, "__call__"):
                item = field(item)
            else:
                item = item[field]
        return item

    def __call__(self, item):
        
        vals = [self.apply_fields(item, f) for f in self.fields]
        return sum(vals) / len(vals)

    def __repr__(self):

        items = []
        for f in self.fields:
            if not isinstance(f, (list, tuple)):
                f = [f]
            items.append(".".join([str(i) for i in f]))

        return "AverageGetters(" + "&".join(items) + ")"
