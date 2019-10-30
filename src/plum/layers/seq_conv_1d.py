from ..types import register, PlumModule, P, HP, props
from .functional import seq_conv1d, dropout


@register("layers.seq_conv_1d")
class SeqConv1D(PlumModule):
    
   
    in_feats = HP(type=props.INTEGER)
    out_feats = HP(type=props.INTEGER)
    kernel_size = HP(type=props.INTEGER)
    batch_first = HP(default=True, type=props.BOOLEAN)
    padding = HP(default=0, type=props.INTEGER)
    dropout = HP(default=0, type=props.REAL, tags=["dropout"])

    weight = P("out_feats", "in_feats", "kernel_size", 
               tags=["weight", "conv_weight", "conv_1d"])
               
    bias = P("out_feats", tags=["bias", "conv_bias", "conv_1d"])

    def forward(self, inputs):
        #batch x out_feats x seq
        output = seq_conv1d(inputs, self.weight, bias=self.bias,
                            padding=self.padding)

        return dropout(output, p=self.dropout, training=self.training)
