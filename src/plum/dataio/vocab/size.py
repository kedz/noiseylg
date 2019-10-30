from ...types import register


@register("dataio.vocab.size")
def size(vocab):
    return len(vocab)
