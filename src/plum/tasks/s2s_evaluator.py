import plum
from plum.types import register, PlumObject, HP, props


@register("plum.tasks.s2s.evaluator")
class S2SEvaluator(PlumObject):

    batches = HP()
    searches = HP(default={})
    metrics = HP(default={})
    loggers = HP(default={})
    loss_function = HP(required=False)
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
        self.preflight_checks(model, env, verbose=verbose)

        if self.loss_function is not None:
            self.loss_function.reset()

        if self.metrics is not None:
            self.metrics.reset()

        self.reset_loggers(self.loggers)
        num_batches = len(self.batches)

        for step, batch in enumerate(self.batches, 1):
           
            forward_state = model(batch)
            if self.loss_function is not None:
                self.loss_function(forward_state, batch)
            if self.metrics is not None:
                self.metrics(forward_state, batch)
            self.apply_loggers(forward_state, batch, self.loggers)

            print("eval: {}/{} loss={:7.6f}".format(
                step, num_batches, self.loss_function.scalar_result()), 
                end="\r" if step < num_batches else "\n", flush=True)
        if self.metrics is not None:
            print(self.metrics.pretty_result())
        result = {
            "loss": {
                "combined": self.loss_function.scalar_result(),
                "detail": self.loss_function.compute(),
            },
            "metrics": self.valid_metrics.compute(),
        }
        
        self.log_results(result)
        print()

        self.close_loggers()

           
 
    def preflight_checks(self, model, env, verbose=False):
        if env["gpu"] > -1:
            if verbose:
                print("Moving model to device: {}".format(env["gpu"]))
            model.cuda(env["gpu"])
            if verbose:
                print("Moving batches to device: {}".format(env["gpu"]))
            self.batches.gpu = env["gpu"]

        model.search_algos.update(self.searches)

#?        if verbose:
#?            print("Logging to tensorboard directory: {}".format(
#?                env["tensorboard_dir"]))
#?        self._tb_writer = SummaryWriter(log_dir=env["tensorboard_dir"])

        log_dir = env["proj_dir"] / "logging.valid" 
        if verbose:
            print("Setting log directory: {}".format(log_dir))
        for logger in self.loggers.values():
            logger.set_log_dir(log_dir)

    def apply_loggers(self, forward_state, batch, loggers):
        for logger in loggers.values():
            logger(forward_state, batch)

    def reset_loggers(self, loggers):
        for logger in loggers.values():
            logger.next_epoch()

    def close_loggers(self):
        for logger in self.loggers.values():
            logger.close()
