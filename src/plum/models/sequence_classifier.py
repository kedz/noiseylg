from ..types import register, PlumModule, HP, SM, LazyDict
from .plum_model import PlumModel


@register("plum.models.sequence_classifier")
class SequenceClassifier(PlumModel):
    
    encoder_inputs = HP(default=["inputs"])
    encoder = SM()
    predictor = SM()

    def __pluminit__(self):
        self.search_algos = {}

    def forward(self, batch):
        encoder_args = [batch[input] for input in self.encoder_inputs]
        encoder_output = self.encoder(*encoder_args)
        output = self.predictor(encoder_output)

        return output
