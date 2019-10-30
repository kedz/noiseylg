local data = import 'PB_DATA.libsonnet';
local vocab = import 'PB_VOCAB.libsonnet';
local pipeline = data.pipeline;

{
    parallel_jsonl(src_path, tgt_path, mmap=false, name=null) : data.parallel(
        [data.jsonl(src_path, mmap=mmap), data.jsonl(tgt_path, mmap=mmap)], 
        name=name,
    ),
    vocab(ds, src_getter, tgt_getter, tgt_at_least=0, tgt_top_k=null) : {
        source: vocab.new(
            "source_vocab",
            ds,
            src_getter,
            start_token="<sos>",
            stop_token="<eos>",
            pad_token="<pad>",
        ),
        target: vocab.new(
            "target_vocab",
            ds,
            tgt_getter,
            start_token="<sos>",
            stop_token="<eos>",
            unknown_token="<unk>",
            pad_token="<pad>",
            at_least=tgt_at_least,
            top_k=tgt_top_k,
        ),
    },
    batches(ds, source, source_vocab, target, target_vocab, 
            target_references=null, batch_size=1, shuffle=true,
            num_workers=1, source_prefix=[0], target_prefix=[1],
            sort=false, sort_key=null, sort_descending=true, 
            extra_pipelines={},
            multiref=false, pad_value=0) :

        if ! multiref then
            data.batches(
                ds, batch_size=batch_size, shuffle=shuffle, 
                num_workers=num_workers,
                sort=sort,
                sort_key=sort_key,
                sort_descending=sort_descending,
                pipelines={
                    source_inputs: source_prefix + source + [
                        data.pipeline.pad_list("<sos>", end=false),
                        data.pipeline.pad_list("<eos>", start=false),
                        data.pipeline.vocab_lookup(source_vocab),
                    ],
                    target_inputs: target_prefix + target + [
                        data.pipeline.pad_list("<sos>", end=false),
                        data.pipeline.vocab_lookup(target_vocab),
                    ],
                    target_outputs: target_prefix + target + [
                        data.pipeline.pad_list("<eos>", start=false),
                        data.pipeline.vocab_lookup(target_vocab),
                    ],
                } + (if target_references != null then 
                    {target_references: target_prefix + target_references} 
                    else {}
                )
                + extra_pipelines,    
                collate_funcs={
                    source_inputs: pipeline.batch_sequence_ndtensor(0, pad_value, 1),
                    target_inputs: pipeline.batch_sequence_ndtensor(0, pad_value, 1),
                    target_outputs: pipeline.batch_sequence_ndtensor(0, pad_value, 1),
                },
        ) else data.batches(
                ds, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers,
                sort=sort,
                sort_key=sort_key,
                sort_descending=sort_descending,
                pipelines={
                    source_inputs: source_prefix + source + [
                        data.pipeline.pad_list("<sos>", end=false),
                        data.pipeline.pad_list("<eos>", start=false),
                        data.pipeline.vocab_lookup(source_vocab),
                    ],
                    target_inputs: target_prefix + [
                        "references", 
                        data.pipeline.aggregate(
                             target + [
                             data.pipeline.pad_list("<sos>", end=false),
                             data.pipeline.vocab_lookup(target_vocab),
                        ]),
                        data.pipeline.batch_sequence_ndtensor(0, pad_value, 1),
                    ],
                    target_outputs: target_prefix + [
                        "references",
                        data.pipeline.aggregate(
                            target + [
                            data.pipeline.pad_list("<eos>", start=false),
                            data.pipeline.vocab_lookup(target_vocab),
                        ]),
                        data.pipeline.batch_sequence_ndtensor(0, pad_value, 1),
                    ],
                    num_references: target_prefix + ["references", pipeline.len()],
                } + (
                    if target_references != null then 
                    {target_references: target_prefix + ["references",
                        data.pipeline.aggregate(target_references)]}
                    else {}
                )
                + extra_pipelines,
                collate_funcs={
                    source_inputs: pipeline.batch_sequence_ndtensor(0, pad_value, 1),
                    target_inputs: {
                        __plum_type__: "dataio.pipeline.batch_variables",
                        pad_batches: true,
                    },
                    target_outputs: {
                        __plum_type__: "dataio.pipeline.batch_variables",
                        pad_batches: true,
                    },
                    num_references: pipeline.long_tensor(),
                },
        ),
    models: import 's2s.models.libsonnet',
    metrics: {
        eval_script(path, search_fields, references_fields) : {
            __plum_type__: "metrics.seq2seq_eval_script",
            path: path,
            search_fields: search_fields,
            references_fields: references_fields,
        },
    },


    evaluator(batches, searches={}, loss_function=null, metrics=null,
              loggers={}, name="s2s_evaluator") : {
        __plum_type__: "plum.tasks.s2s.evaluator",
        __plum_program__: name,
        batches: batches,
        searches: searches,
        loggers: loggers
        
    } + (if loss_function != null then {loss_function: loss_function} else {})
    + (if metrics != null then {metrics: metrics} else {}),
}
