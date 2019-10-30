from ..types import register, PlumModule, HP, P, props
from .activation_function import ActivationFunction
import torch
from .functional import linear, dropout


@register("layers.fully_connected")
class FullyConnected(PlumModule):

    in_feats = HP(type=props.POSITIVE)
    out_feats = HP(type=props.POSITIVE)
    has_bias = HP(default=True)
    dropout = HP(default=0, type=props.NON_NEGATIVE, tags=["dropout"])
    in_dim = HP(default=None, required=False)
    activation = HP(default=ActivationFunction(name="tanh"))

    weight = P("out_feats", "in_feats", 
               tags=["weight", "fully_connected"])
    bias = P("out_feats", conditional="has_bias", 
             tags=["bias", "fully_connected"])

    def forward(self, inputs):
        preact = linear(inputs, self.weight, self.bias, self.in_dim)
        act = self.activation(preact)
        output = dropout(act, p=self.dropout, training=self.training)
        return output
