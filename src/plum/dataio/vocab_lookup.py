from ..types import register, PlumObject, HP, props
import torch


@register("dataio.pipeline.vocab_lookup")
class VocabLookup(PlumObject):

    vocab = HP()

    def __call__(self, item):
        if not isinstance(item, (list, tuple)):
            item = [item]
        return torch.LongTensor([self.vocab[token] for token in item])
