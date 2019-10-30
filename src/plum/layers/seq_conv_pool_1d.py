from ..types import register, PlumModule, P, HP, props
from torch.nn.functional import conv1d, dropout
from .seq_conv_1d import SeqConv1D
from .seq_pool_1d import SeqPool1D
from .activation_function import ActivationFunction


@register("layers.seq_conv_pool_1d")
class SeqConvPool1D(PlumModule):
    
    in_feats = HP(type=props.INTEGER)
    out_feats = HP(type=props.INTEGER)
    kernel_size = HP(type=props.INTEGER)
    batch_first = HP(default=True, type=props.BOOLEAN)
    padding = HP(default=0, type=props.INTEGER)
    dropout = HP(default=0, type=props.REAL, tags=["dropout"])
    activation = HP(
        default="relu",
        type=props.Choice(["relu", "tanh", "sigmoid", "identity"]))

    def __pluminit__(self, in_feats, out_feats, kernel_size, batch_first,
                     padding, activation):
        self.conv = SeqConv1D(in_feats=in_feats, out_feats=out_feats,
                              kernel_size=kernel_size, batch_first=batch_first,
                              padding=padding)
        self.pool = SeqPool1D() #batch_dim=0, seq_dim=2)
        self.act = ActivationFunction(name=activation)

    def forward(self, inputs):
        output = self.act(self.pool(self.conv(inputs)).squeeze(2))
        return dropout(output, p=self.dropout, training=self.training)
