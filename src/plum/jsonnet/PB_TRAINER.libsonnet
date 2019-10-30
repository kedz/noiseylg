local trainer(model, tr_batches, va_batches, optimizer, loss_func, 
              train_metrics=null, valid_metrics=null, max_epochs=10,
              checkpoints=null, searches={}, train_loggers={},
              valid_loggers={}, warm_start=false,
              name="train", valid_metrics_start=0) = {
    __plum_type__: "trainer.basic_trainer",
    __plum_program__: name,
    model: model,
    train_batches: tr_batches,
    valid_batches: va_batches,
    optimizer: optimizer,
    loss_function: loss_func,
    warm_start: warm_start,
    train_metrics: train_metrics,
    valid_metrics: valid_metrics,
    max_epochs: max_epochs,
    searches: searches,
    valid_metrics_start: valid_metrics_start, 
    train_loggers: train_loggers,
    valid_loggers: valid_loggers,
} + if checkpoints != null then {checkpoints: checkpoints} else {};

trainer
