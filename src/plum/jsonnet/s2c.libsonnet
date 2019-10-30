local data = import 'PB_DATA.libsonnet';
local vocab = import 'PB_VOCAB.libsonnet';
local pipeline = data.pipeline;

{
    parallel_jsonl(src_path, tgt_path, name=null) : data.parallel(
        [data.jsonl(src_path), data.jsonl(tgt_path)], name=name,
    ),
    
#    batches(ds, source, source_vocab, target, target_vocab, 
#            target_references=null, batch_size=1, shuffle=true,
#            num_workers=1,
#            multiref=false) :
#
#        if ! multiref then
#            data.batches(
#                ds, batch_size=batch_size, shuffle=shuffle, 
#                num_workers=num_workers,
#                pipelines={
#                    source_inputs: [0] + source + [
#                        data.pipeline.pad_list("<sos>", end=false),
#                        data.pipeline.pad_list("<eos>", start=false),
#                        data.pipeline.vocab_lookup(source_vocab),
#                    ],
#                    target_inputs: [1] + target + [
#                        data.pipeline.pad_list("<sos>", end=false),
#                        data.pipeline.vocab_lookup(target_vocab),
#                    ],
#                    target_outputs: [1] + target + [
#                        data.pipeline.pad_list("<eos>", start=false),
#                        data.pipeline.vocab_lookup(target_vocab),
#                    ],
#                    #target_references: [1, "reference_string"],
#                } + if target_references != null then 
#                    {target_references: [1] + target_references} 
#                    else {},
#                collate_funcs={
#                    source_inputs: pipeline.batch_sequence_ndtensor(0, -1, 1),
#                    target_inputs: pipeline.batch_sequence_ndtensor(0, 0, 1),
#                    target_outputs: pipeline.batch_sequence_ndtensor(0, 0, 1),
#                },
#        ) else data.batches(
#                ds, batch_size=batch_size, shuffle=shuffle,
#                num_workers=num_workers,
#                pipelines={
#                    source_inputs: [0] + source + [
#                        data.pipeline.pad_list("<sos>", end=false),
#                        data.pipeline.pad_list("<eos>", start=false),
#                        data.pipeline.vocab_lookup(source_vocab),
#                    ],
#                    target_inputs: [
#                        1, "references", 
#                        data.pipeline.aggregate(
#                             target + [
#                             data.pipeline.pad_list("<sos>", end=false),
#                             data.pipeline.vocab_lookup(target_vocab),
#                        ]),
#                        data.pipeline.batch_sequence_ndtensor(0, 0, 1),
#                    ],
#                    target_outputs: [
#                        1, "references",
#                        data.pipeline.aggregate(
#                            target + [
#                            data.pipeline.pad_list("<eos>", start=false),
#                            data.pipeline.vocab_lookup(target_vocab),
#                        ]),
#                        data.pipeline.batch_sequence_ndtensor(0, 0, 1),
#                    ],
#                    num_references: [1, "references", pipeline.len()],
#                } + if target_references != null then 
#                    {target_references: [1, "references",
#                        data.pipeline.aggregate(target_references)]}
#                    else {},
#                collate_funcs={
#                    source_inputs: pipeline.batch_sequence_ndtensor(0, -1, 1),
#                    target_inputs: [
#                        pipeline.pad_dim2max(0, pad=0), 
#                        pipeline.pad_dim2max(1, pad=0),
#                        pipeline.cat(1),
#                    ],
#                    target_outputs: [
#                        pipeline.pad_dim2max(0, pad=0), 
#                        pipeline.pad_dim2max(1, pad=0),
#                        pipeline.cat(1),
#                    ],
#                    num_references: pipeline.long_tensor(),
#                },
#        ),
    models: import 's2c.models.libsonnet',
}
