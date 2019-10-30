from ..types import register, PlumObject, HP, props
from ..vocab import Vocab

from collections import Counter


@register("dataio.vocab_reader")
class VocabReader(PlumObject):

    dataset = HP()
    pipeline = HP()
    start_token = HP(required=False, type=props.STRING)
    stop_token = HP(required=False, type=props.STRING)
    unknown_token = HP(required=False, type=props.STRING)
    pad_token = HP(required=False, type=props.STRING)
    top_k = HP(required=False, type=props.INTEGER)
    at_least = HP(default=0, type=props.INTEGER)
    
    def __new__(cls, *args, **kwargs):

        counts = Counter()
        for item in kwargs["dataset"]:
            for pipe in kwargs["pipeline"]:
                if hasattr(pipe, "__call__"):
                    item = pipe(item)
                else:
                    item = item[pipe]
            if not isinstance(item, (list, tuple)):
                item = [item]
            counts.update(item)

        return Vocab.from_counts(
            counts, 
            start=kwargs.get("start_token", None),  
            stop=kwargs.get("stop_token", None), 
            unk=kwargs.get("unknown_token", None),
            pad=kwargs.get("pad_token", None),
            at_least=kwargs.get("at_least", None),
            top_k=kwargs.get("top_k", 0)
        )
