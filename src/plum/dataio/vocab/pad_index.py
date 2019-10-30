from ...types import register


@register("dataio.vocab.pad_index")
def pad_index(vocab):
    return vocab.pad_index
