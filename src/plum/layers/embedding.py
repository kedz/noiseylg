from ..types import register, PlumModule, P, HP, props
from .functional import embedding, dropout


@register("layers.embedding")
class Embedding(PlumModule):

    in_feats =  HP(type=props.INTEGER)
    out_feats = HP(type=props.INTEGER)
    pad_index = HP(required=False)
    dropout = HP(default=0.0, tags=["dropout", "embedding_dropout"])
#    feature_dropout = HP(default=0.0, tags=["feature_dropout"])

    weight = P("in_feats", "out_feats", tags=["weight", "embedding"])
# TODO add callback to fix weightes even after init, pad weight should be 0.

    def forward(self, inputs):
        result = embedding(inputs, self.weight, padding_idx=self.pad_index)
        output = dropout(result, p=self.dropout, training=self.training)
        return output
