local PM = import 'PM.libsonnet';

local root_dir = "../";

local target_field = "customer_rating";

local train_ds = PM.s2c.parallel_jsonl(
    root_dir + "data/e2e/e2e.train.descs.v5.delex.jsonl",
    root_dir + "data/e2e/e2e.train.mrs.v5.delex.jsonl",
    name="e2e.train");

local valid_ds = PM.s2c.parallel_jsonl(
    root_dir + "data/e2e/e2e.valid.descs.v5.delex.jsonl",
    root_dir + "data/e2e/e2e.valid.mrs.v5.delex.jsonl",
    name="e2e.valid");

local input_vocab = PM.vocab.load(
    "base_generators/delex/vocabs/target_vocab.pth",
    name="input_vocab");

local label_vocab = PM.vocab.new(target_field, train_ds,
                                 [1, "labels", target_field]);

local pipelines = {
    inputs: [
        0, "sequence", "tokens", 
        PM.data.pipeline.pad_list("<sos>", end=false),
        PM.data.pipeline.pad_list("<eos>", start=false),
        PM.data.pipeline.vocab_lookup(input_vocab),
    ],
    reference_input: [0, "reference_string"],
    reference_output: [1, "labels", target_field],
    targets: [
        1, "labels", target_field, 
        PM.data.pipeline.vocab_lookup(label_vocab)
    ],
};

local collate_funcs = {
    inputs: PM.data.pipeline.batch_sequence_ndtensor(0, 0, 0),
    targets: PM.data.pipeline.batch_flat(),
};

local train_batches = PM.data.batches(
    train_ds, 
    batch_size=128,
    pipelines=pipelines,
    collate_funcs=collate_funcs,
    num_workers=4,
);

local valid_batches = PM.data.batches(
    valid_ds, 
    batch_size=128,
    pipelines=pipelines,
    collate_funcs=collate_funcs,
    num_workers=4,
);

local model = PM.s2c.models.cnn(50, input_vocab, label_vocab,
                                filter_widths=[1,2,3],
                                features=[50, 50, 50]);

local checkpoints = {
    __plum_type__: "checkpoints.topk",
    k: 3,
    criterion: ["valid", "metrics", "average", "fscore"],
    min_criterion: false,
};

local loss_function = PM.loss.cross_entropy(
    labels_field="targets",
);


local train_loggers = {
    model_output: {
        __plum_type__: "plum.loggers.classification_logger",
        file_prefix: "model_output",
        input_fields: "reference_input",
        output_fields: "output",
        target_fields: "reference_output",
        vocab: label_vocab,
        log_every: 50,
    },
};

local valid_loggers = {
    model_output: {
        __plum_type__: "plum.loggers.classification_logger",
        file_prefix: "model_output",
        input_fields: "reference_input",
        output_fields: "output",
        target_fields: "reference_output",
        vocab: label_vocab,
    },
};

local metrics = PM.metrics.class_prf(vocab=label_vocab);

[
    PM.trainer(
        model, train_batches, valid_batches,
        PM.optim.sgd(lr=0.25, weight_decay=0.0001),
        loss_function, 
        train_metrics=metrics,
        valid_metrics=metrics,
        max_epochs=30,
        train_loggers=train_loggers,
        valid_loggers=valid_loggers,
        checkpoints=checkpoints,
    ),
]
