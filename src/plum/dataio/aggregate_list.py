from ..types import register, PlumObject, HP, props


@register("dataio.pipeline.aggregate_list")
class AggregateList(PlumObject):

    fields = HP()

    def __call__(self, list_item):
        if not isinstance(list_item, (list, tuple)):
            raise ValueError("item is not a list or tuple.")

        result = []

        for item in list_item:
            processed_item = item
            for pipe in self.fields:
                if hasattr(pipe, "__call__"):
                    processed_item = pipe(processed_item)
                else:
                    processed_item = processed_item[pipe]
            result.append(processed_item)

        return result
