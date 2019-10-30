local PM = import 'PM.libsonnet';

local root_dir = "../";

local train_ds = PM.s2s.parallel_jsonl(
    root_dir + "data/e2e/e2e.train.mrs.v6.lex.jsonl",
    root_dir + "data/e2e/e2e.train.descs.v6.lex.jsonl",
    name="e2e.train");

local valid_ds = PM.s2s.parallel_jsonl(
    root_dir + "data/e2e/e2e.valid.multiref.mrs.v6.lex.jsonl",
    root_dir + "data/e2e/e2e.valid.multiref.descs.v6.lex.jsonl",
    name="e2e.valid");

local vocab = PM.s2s.vocab(
    train_ds,
    [0, "sequence", "mrs"], 
    [1, "sequence", "tokens"],
    tgt_at_least=10,
);

local train_batches = PM.s2s.batches(
    train_ds,
    ["sequence", "mrs"], vocab["source"],
    ["sequence", "tokens"], vocab["target"],
    target_references=["reference_string"],
    batch_size=128,
    num_workers=4,
);

local valid_batches = PM.s2s.batches(
    valid_ds,
    ["sequence", "mrs"], vocab["source"],
    ["sequence", "tokens"], vocab["target"],
    target_references=["reference_string"],
    batch_size=8,
    multiref=true,
    num_workers=4,
    extra_pipelines={labels: [0, "labels"]},
);

local model = PM.s2s.models.rnn(512, vocab["source"], vocab["target"]);

local metrics = PM.metrics.dict({
    greedy: {
        __plum_type__: "d2t.metrics.e2e_eval_script",
        path: root_dir + "eval_scripts/eval.py", 
        search_fields: ["search", "greedy"], 
        references_fields: "target_references",
        labels_fields: "labels",
    },
    beam8: {
        __plum_type__: "d2t.metrics.e2e_eval_script",
        path: root_dir + "eval_scripts/eval.py", 
        search_fields: ["search", "beam8"], 
        references_fields: "target_references",
        labels_fields: "labels",
    },

});


local checkpoints = {
    __plum_type__: "checkpoints.topk",
    k: 10,
    criterion: ["valid", "metrics", "beam8", "BLEU"],
    min_criterion: false,
};

local loss_function = PM.loss.cross_entropy(
    labels_field="target_outputs",
    padding_index=PM.vocab.pad_index(vocab["target"]),
);

local searches = {
    greedy: {
        __plum_type__: "seq2seq.search.greedy",
        max_steps: 200,
        vocab: vocab["target"],
    },
    beam8: {
        __plum_type__: "seq2seq.search.beam",
        max_steps: 200,
        beam_size: 8,
        vocab: vocab["target"],
    },
};

local loggers = {
    "greedy": {
        __plum_type__: "fg.loggers.e2e_search_logger",
        file_prefix: "greedy",
        search_fields: ["search", "greedy"],
        input_fields: "labels",
        reference_fields: "target_references",
    },
    "beam8": {
        __plum_type__: "fg.loggers.e2e_search_logger",
        file_prefix: "greedy",
        search_fields: ["search", "beam8"],
        input_fields: "labels",
        reference_fields: "target_references",
    },

};

[
    PM.trainer(
        model, train_batches, valid_batches,
        PM.optim.sgd(lr=0.25, weight_decay=0.0001),
        loss_function, 
        valid_metrics=metrics,
        valid_loggers=loggers,
        max_epochs=500,
        checkpoints=checkpoints,
        searches=searches,
    ),
    {
        __plum_type__: "d2t.e2e_systematic_generation",
        __plum_program__: "systematic_generation_beam8_mr3",
        mr_size: 3,
		batch_size: 32,
		beam_size: 8,
		source_vocab: vocab["source"],
		target_vocab: vocab["target"],
		filename: "base.lex.beam8.mr3.jsonl"
    },
    {
        __plum_type__: "d2t.e2e_predict",
        __plum_program__: "generate_valid_beam8",
		beam_size: 8,
		source_vocab: vocab["source"],
		target_vocab: vocab["target"],
		input_path: root_dir + "data/e2e/e2e.valid.multiref.mrs.v6.lex.jsonl", 
		filename: "base.lex.beam8.valid.jsonl"
    },
    {
        __plum_type__: "d2t.e2e_predict",
        __plum_program__: "generate_test_beam8",
		beam_size: 8,
		source_vocab: vocab["source"],
		target_vocab: vocab["target"],
		input_path: root_dir + "data/e2e/e2e.test.mrs.v6.lex.jsonl", 
		filename: "base.lex.beam8.test.jsonl"
    },
    {
        __plum_type__: "d2t.e2e_predict",
        __plum_program__: "generate_test_greedy",
		beam_size: 1,
		source_vocab: vocab["source"],
		target_vocab: vocab["target"],
		input_path: root_dir + "data/e2e/e2e.test.mrs.v6.lex.jsonl", 
		filename: "base.lex.greedy.test.jsonl"
    },

]
