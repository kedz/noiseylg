from ..types import register, PlumModule, HP, P, props, LazyDict
import torch
from .functional import linear


@register("layers.linear_predictor")
class LinearPredictor(PlumModule):

    in_feats = HP(type=props.POSITIVE)
    num_classes = HP(type=props.POSITIVE)
    has_bias = HP(default=True)

    weight = P("num_classes", "in_feats", 
               tags=["weight", "linear_predictor"])
    bias = P("num_classes", conditional="has_bias", 
             tags=["bias", "linear_predictor"])

    def _curry_output(self, logits):
        def get_output():
            return logits.argmax(-1)
        return get_output

    def _curry_probs(self, logits):
        def get_probs():
            return logits.softmax(dim=-1)
        return get_probs

    def _curry_log_probs(self, logits):
        def get_log_probs():
            return logits.log_softmax(dim=-1)
        return get_log_probs

    def forward(self, inputs):

        logits = linear(inputs, self.weight, self.bias)
        result = LazyDict()
        result["target_logits"] = logits
        result.lazy_set("output", self._curry_output(logits))
        result.lazy_set("probs", self._curry_probs(logits))
        result.lazy_set("log_probs", self._curry_log_probs(logits))



        return result
