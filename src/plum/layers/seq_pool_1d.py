from ..types import register, PlumModule, P, HP, props
from .functional import seq_max_pool1d


@register("layers.seq_pool_1d")
class SeqPool1D(PlumModule):

#    batch_dim = HP(type=props.INTEGER)
#    seq_dim = HP(type=props.INTEGER)
    
    def forward(self, inputs):
        return seq_max_pool1d(inputs)
     
