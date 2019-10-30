from ..types import register, PlumObject, HP, SM, props
from ..metrics import MetricDict
#from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import plum
from pprint import pprint


@register("eval.basic_eval")
class BasicEval(PlumObject):

    loss_function = HP(required=False)
    metrics = HP(required=False)
    batches = HP()
    searches = HP(default={}, required=False)
    checkpoint = HP(required=False)

    def _get_default_checkpoint(self, env):
        for ckpt, md in env["checkpoints"].items():
            if md.get("default", False):
                return ckpt
        return ckpt

    def run(self, env, verbose=False):
        if self.checkpoint is None:
            ckpt = self._get_default_checkpoint(env)
        else:
            ckpt = self.checkpoint
        if ckpt is None:
            raise RuntimeError("No checkpoints found!")
        
        ckpt_path = env["checkpoints"][ckpt]["path"]
        if verbose:
            print("Reading checkpoint from {}".format(ckpt_path))
        model = plum.load(ckpt_path).eval()
        model.search_algos.update(self.searches)

        if env["gpu"] > -1:
            model.cuda(env["gpu"])
            self.batches.gpu = env["gpu"]
        
        if self.loss_function:
            self.loss_function.reset()

        if self.metrics:
            self.metrics.reset()

        for batch in self.batches:
            
            state = model(batch)
            
            if self.loss_function:
                self.loss_function(state, batch)
            
            if self.metrics:
                self.metrics(state, batch)
            
        print("loss")
        pprint(self.loss_function.compute())
        print()

        if self.metrics:
            print("metrics")
            pprint(self.metrics.compute())
            print()
