from ..types import register, PlumObject, HP, props
import json
from warnings import warn


@register("checkpoints.none")
class NoCheckpoint(PlumObject):

    def __pluminit__(self):
        self._epoch = 0
        self._dir = None
        self._results_path = None
        self.verbose = False

    def set_dir(self, path):
        self._dir = path
        self._dir.mkdir(exist_ok=True, parents=True)
        self._results_path = self._dir / "results.jsonl"
    

    def __call__(self, results, model):

        self._epoch += 1
        if self.verbose:
            warn("Not saving checkpoints.")
        with open(self._results_path, "a") as fp:
            print(json.dumps(results), file=fp)
