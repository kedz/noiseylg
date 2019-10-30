from ..types import register, PlumObject, HP, props, Variable
import plum


@register("dataio.pipeline.batch_variables")
class BatchVariables(PlumObject):

    pad_batches = HP(default=False, type=props.BOOLEAN)

    def __call__(self, items):

        bd0 = items[0].batch_dim
        sd0 = items[0].length_dim
        pv0 = items[0].pad_value
        for item in items[1:]:
            assert bd0 == item.batch_dim
            assert sd0 == item.length_dim
            assert pv0 == item.pad_value

        if self.pad_batches:
            batch_sizes = [item.batch_size for item in items]
            max_batch_size = max(batch_sizes)
            items = [item.pad_batch_dim(max_batch_size - item.batch_size)
                     for item in items]
        lengths = plum.cat([item.lengths for item in items])
        max_len = lengths.max()
        items = [item.pad_length_dim(max_len - item.lengths.max())
                 for item in items] 

        data = plum.cat([item.data for item in items], dim=bd0)

        return Variable(data, lengths=lengths, batch_dim=bd0, length_dim=sd0,
                        pad_value=pv0)
