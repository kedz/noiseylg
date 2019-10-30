from ..types import register, PlumObject, HP, props
import json


@register("checkpoints.topk")
class TopKCheckpoint(PlumObject):

    k = HP(default=1, type=props.INTEGER)
    criterion = HP(default=["valid", "loss", "combined"])
    min_criterion = HP(default=True, type=props.BOOLEAN)

    def __pluminit__(self):
        self._epoch = 0
        self._dir = None
        self._results_path = None
        self._top_k = []
        self.verbose = False

    def set_dir(self, path):
        self._dir = path
        self._dir.mkdir(exist_ok=True, parents=True)
        self._results_path = self._dir / "results.jsonl"
        self._meta_path = self._dir / "ckpt.metadata.json"
 
    def get_criterion(self, results):
        c = results
        criterion = self.criterion
        if not isinstance(criterion, (list, tuple)):
            criterion = [criterion]
        for field in criterion:
            if hasattr(field, "__call__"):
                c = field(c)
            else:
                c = c[field]
        return c

    def __call__(self, results, model):
        self._epoch += 1

        with open(self._results_path, "a") as fp:
            print(json.dumps(results), file=fp)

        ckpt_path = self._dir / "model.ckpt.{}.pth".format(self._epoch)
        crit = self.get_criterion(results)
        cmp = min_cmp if self.min_criterion else max_cmp

        if len(self._top_k) < self.k:
            self._top_k.append((crit, ckpt_path))
            self._top_k.sort(key=lambda x: x[0], 
                             reverse=not self.min_criterion)
            if self.verbose:
                print("Saving model: {}".format(ckpt_path))
            model.save(ckpt_path)
            self._write_meta()

        elif any([cmp(crit, old[0]) for old in self._top_k]):
            if self.verbose:
                print("Saving model: {}".format(ckpt_path))
            model.save(ckpt_path)
            self._top_k.append((crit, ckpt_path))
            self._top_k.sort(key=lambda x: x[0], 
                             reverse=not self.min_criterion)
            _, delete_path = self._top_k.pop(-1)
            delete_path.unlink()
            self._write_meta()

    def _write_meta(self):
        crit = self.criterion
        if not isinstance(crit, (list, tuple)):
            crit = [crit]
        meta = {
            "criterion": ".".join([str(x) for x in crit]),
            "min_criterion": self.min_criterion,
            "optimal_checkpoint": self._top_k[0][1].name,
            "optimal_criterion": self._top_k[0][0],
            "checkpoint_manifest": [
                {"checkpoint": x[1].name, "criterion": x[0]}
                for x in self._top_k
            ]
        }
        self._meta_path.write_text(json.dumps(meta, indent=4, sort_keys=True))

def min_cmp(new, old):
    return new < old

def max_cmp(new, old):
    return new > old
