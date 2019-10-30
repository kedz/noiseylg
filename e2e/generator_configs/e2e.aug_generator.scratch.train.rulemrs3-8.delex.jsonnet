local PM = import 'PM.libsonnet';


local synth_data = [
    PM.data.jsonl(
        "samples/base.delex.rules.mr3.max25k.jsonl",
        name="delex.rule.mr3", mmap=true),
    PM.data.jsonl(
        "samples/base.delex.rules.mr4.max25k.jsonl",
        name="delex.rule.mr4", mmap=true),
    PM.data.jsonl(
        "samples/base.delex.rules.mr5.max25k.jsonl",
        name="delex.rule.mr5", mmap=true),
    PM.data.jsonl(
        "samples/base.delex.rules.mr6.max25k.jsonl",
        name="delex.rule.mr6", mmap=true),
    PM.data.jsonl(
        "samples/base.delex.rules.mr7.max25k.jsonl",
        name="delex.rule.mr7", mmap=true),
    PM.data.jsonl(
        "samples/base.delex.rules.mr8.max25k.jsonl",
        name="delex.rule.mr8", mmap=true),
];

local train_src = PM.data.jsonl(
    "../data/e2e/e2e.train.mrs.v5.delex.jsonl", name="e2e.train.src.orig",
    mmap=true);
local train_tgt = PM.data.jsonl(
    "../data/e2e/e2e.train.descs.v5.delex.jsonl", name="e2e.train.tgt.orig",
    mmap=true);


local train_ds = PM.data.parallel(
    [
        PM.data.stack_ds([train_src] + synth_data, 'e2e.train.src.aug'),
        PM.data.stack_ds([train_tgt] + synth_data, 'e2e.train.tgt.aug'),
    ],
    name="e2e.train",
);

local valid_ds = PM.s2s.parallel_jsonl(
    "../data/e2e/e2e.valid.multiref.mrs.v5.delex.jsonl",
    "../data/e2e/e2e.valid.multiref.descs.v5.delex.jsonl",
    name="e2e.valid");

local vocab = {
    "source": PM.vocab.load("base_generators/delex/vocabs/source_vocab.pth", name="source"),
    "target": PM.vocab.load("base_generators/delex/vocabs/target_vocab.pth", name="target"),
};

local train_batches = PM.s2s.batches(
    train_ds,
    ["sequence", "mrs"], vocab["source"],
    ["sequence", "tokens"], vocab["target"],
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
        path: "../eval_scripts/eval.py", 
        search_fields: ["search", "greedy"], 
        references_fields: "target_references",
        labels_fields: "labels",
    },
    beam8: {
        __plum_type__: "d2t.metrics.e2e_eval_script",
        path: "../eval_scripts/eval.py", 
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
        file_prefix: "beam8",
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
        max_epochs=50,
        checkpoints=checkpoints,
        searches=searches,
    ),
    {
        __plum_type__: "d2t.e2e_predict",
        __plum_program__: "generate_valid_beam8",
		beam_size: 8,
		source_vocab: vocab["source"],
		target_vocab: vocab["target"],
		input_path: "../data/e2e/e2e.valid.multiref.mrs.v5.delex.jsonl", 
		filename: "aug.rule.delex.beam8.valid.jsonl"
    },
    {
        __plum_type__: "d2t.e2e_predict",
        __plum_program__: "generate_test_beam8",
		beam_size: 8,
		source_vocab: vocab["source"],
		target_vocab: vocab["target"],
		input_path: "../data/e2e/e2e.test.mrs.v5.delex.jsonl", 
		filename: "aug.rule.delex.beam8.test.jsonl",
        delex: true,
    },
    {
        __plum_type__: "d2t.e2e_predict",
        __plum_program__: "generate_test_greedy",
		beam_size: 1,
		source_vocab: vocab["source"],
		target_vocab: vocab["target"],
		input_path: "../data/e2e/e2e.test.mrs.v5.delex.jsonl", 
		filename: "aug.rule.delex.greedy.test.jsonl",
        delex: true,
    },
] + [

    {
        __plum_type__: "d2t.e2e_systematic_generation",
        __plum_program__: "systematic_generation_beam8_mr" + mrsize,
        mr_size: mrsize,
		beam_size: 8,
        batch_size: 128,
		source_vocab: vocab["source"],
		target_vocab: vocab["target"],
		filename: "aug.rule.delex.beam8.mr" + mrsize + ".jsonl",
        delex: true,
    },
    for mrsize in [3,4,5,6,7,8]

]
