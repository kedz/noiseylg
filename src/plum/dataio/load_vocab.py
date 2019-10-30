from ..types import register, PlumObject, HP, props
from ..vocab import Vocab
import plum

@register("dataio.load_vocab")
class LoadVocab(PlumObject):

    path = HP(type=props.EXISTING_PATH)

    def __new__(cls, *args, **kwargs):
        path = kwargs["path"]
        return plum.load(path)
        
