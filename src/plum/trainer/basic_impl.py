from ..types import register, PlumObject, HP, SM, props
from ..metrics import MetricDict
from ..checkpoints import TopKCheckpoint
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


@register("trainer.basic_trainer")
class BasicTrainer(PlumObject):
    
    model = HP()
    train_batches = HP() 
    valid_batches = HP()
    optimizer = HP()
    loss_function = HP()
    train_metrics = HP(default=MetricDict(metrics={}))
    valid_metrics = HP(default=MetricDict(metrics={}))
    max_epochs = HP(default=10, type=props.INTEGER)
    checkpoints = HP(default=TopKCheckpoint(k=3), required=False)
    searches = HP(default={}, required=False)
    train_loggers = HP(default={}, required=False)
    valid_loggers = HP(default={}, required=False)
    warm_start = HP(default=False, required=False, type=props.BOOLEAN)
    valid_metrics_start = HP(default=0, required=False, type=props.INTEGER)

    def __pluminit__(self):
        self._epoch = 0
        self._tb_writer = None

    def run(self, env, verbose=False):
        self.preflight_checks(env, verbose=verbose)

        if verbose:
            print("Preflight checks complete. Have a nice trip!")
       
        best_valid_loss = float("inf")
        for epoch in range(1, self.max_epochs + 1):
            self._epoch += 1

            self.train_epoch()
            
            result = {
                "train": {
                    "loss": {
                        "combined": self.loss_function.scalar_result(),
                        "detail": self.loss_function.compute(),
                    },
                    "metrics": self.train_metrics.compute(),
                },
            }

            self.valid_epoch()
            result["valid"] = {
                "loss": {
                    "combined": self.loss_function.scalar_result(),
                    "detail": self.loss_function.compute(),
                },
            }
            if self._epoch >= self.valid_metrics_start:
                result["valid"]["metrics"] = self.valid_metrics.compute()
            
            self.log_results(result)

            if self._epoch >= self.valid_metrics_start:
                self.checkpoints(result, self.model)
            print()

        self._tb_writer.close()
        self.close_loggers()

    def preflight_checks(self, env, verbose=False):

        checkpoint_dir = env["proj_dir"] / "model_checkpoints"
        if verbose:
            print("Setting checkpoint directory: {}".format(checkpoint_dir))
        self.checkpoints.set_dir(checkpoint_dir)
        self.checkpoints.verbose = True

        if self.warm_start:
            if verbose:
                print("Preflight Check: warm_start enabled, "
                      "skipping model init.")
        else:
            if verbose:
                print("Preflight Check: initializing model")
            self.model.initialize_parameters(verbose=verbose)

        if env["gpu"] > -1:
            if verbose:
                print("Moving model to device: {}".format(env["gpu"]))
            self.model.cuda(env["gpu"])
            if verbose:
                print("Moving batches to device: {}".format(env["gpu"]))
            self.train_batches.gpu = env["gpu"]
            self.valid_batches.gpu = env["gpu"]

        if verbose:
            print("Preflight Check: initializing optimizer")
        self.optimizer.setup_optimizer(self, verbose=verbose)

        self.model.search_algos.update(self.searches)

        if verbose:
            print("Logging to tensorboard directory: {}".format(
                env["tensorboard_dir"]))
        self._tb_writer = SummaryWriter(log_dir=env["tensorboard_dir"])

        train_log_dir = env["proj_dir"] / "logging.train" 
        if verbose:
            print("Setting training log directory: {}".format(train_log_dir))
        for logger in self.train_loggers.values():
            logger.set_log_dir(train_log_dir)

        valid_log_dir = env["proj_dir"] / "logging.valid" 
        if verbose:
            print("Setting validation log directory: {}".format(valid_log_dir))
        for logger in self.valid_loggers.values():
            logger.set_log_dir(valid_log_dir)

    def train_epoch(self):
        
        self.model.train()
        self.loss_function.reset()
        self.train_metrics.reset()
        self.reset_loggers(self.train_loggers)
        num_batches = len(self.train_batches)

        for step, batch in enumerate(self.train_batches, 1):
           
            self.optimizer.zero_grad()
            forward_state = self.model(batch)
            loss = self.loss_function(forward_state, batch)
            loss.backward()
            self.train_metrics(forward_state, batch)
            self.optimizer.step()
            self.apply_loggers(forward_state, batch, self.train_loggers)

            print("train epoch {}: {}/{} loss={:7.6f}".format(
                self._epoch,
                step, num_batches, self.loss_function.scalar_result()), 
                end="\r" if step < num_batches else "\n", flush=True)
        print(self.train_metrics.pretty_result())

    def valid_epoch(self):
        
        self.model.eval()
        self.loss_function.reset()
        self.valid_metrics.reset()
        self.reset_loggers(self.valid_loggers)
        num_batches = len(self.valid_batches)

        for step, batch in enumerate(self.valid_batches, 1):
           
            forward_state = self.model(batch)
            self.loss_function(forward_state, batch)
            if self._epoch >= self.valid_metrics_start:
                self.valid_metrics(forward_state, batch)
            self.apply_loggers(forward_state, batch, self.valid_loggers)

            print("valid epoch {}: {}/{} loss={:7.6f}".format(
                self._epoch,
                step, num_batches, self.loss_function.scalar_result()), 
                end="\r" if step < num_batches else "\n", flush=True)
        if self._epoch >= self.valid_metrics_start:
            print(self.valid_metrics.pretty_result())
            
    def _log_helper(self, prefix, results):
        if isinstance(results, dict):
            for key, value in results.items():
                self._log_helper(prefix + "/" + key, value)
        else:
            self._tb_writer.add_scalar(prefix[1:], results, self._epoch) 
                 
    def log_results(self, results):
        self._log_helper("", results) 
        for writer in self._tb_writer.all_writers.values():
            writer.flush()

    def apply_loggers(self, forward_state, batch, loggers):
        for logger in loggers.values():
            logger(forward_state, batch)

    def reset_loggers(self, loggers):
        for logger in loggers.values():
            logger.next_epoch()

    def close_loggers(self):
        for logger in self.train_loggers.values():
            logger.close()
        for logger in self.valid_loggers.values():
            logger.close()
