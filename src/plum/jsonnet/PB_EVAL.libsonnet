local eval(model_name, test_batches, loss_function=null, metrics=null,
           searches={},
           name="eval") = {
    __plum_type__: "eval.basic_eval",
    __plum_program__: name,
    batches: test_batches,
    loss_function: loss_function,
    searches: searches,
    metrics: metrics,
};

eval
