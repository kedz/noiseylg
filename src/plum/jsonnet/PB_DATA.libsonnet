

local data = {
    csv(path, header=true, sep=",", name=null): {
        __plum_type__: "dataio.csv",
        __plum_datasource__: if name != null then name else path,
        path: path,
        header: header,
        sep: sep,
    },

    jsonl(path, mmap=false, name=null): {

        __plum_type__: if mmap then "dataio.mmap_jsonl" else "dataio.jsonl",
        __plum_datasource__: if name != null then name else path,
        path: path,
    },

    stack_ds(datasources, name) : {
        __plum_type__: "dataio.stack_ds",
        __plum_datasource__: name,
        datasources: datasources,
    },
    parallel(datasources, name=null): {
        __plum_type__: "dataio.parallel_datasources",
        __plum_datasource__: if name != null then name else std.join(
            ":", [ds.__plum_datasource__ for ds in datasources]),
        datasources: datasources 
    },
    
    batches(ds, batch_size=32, num_workers=1, shuffle=true, pipelines={},
            collate_funcs={}, sort=false, sort_key=null,
            sort_descending=true): {
        __plum_type__: "dataio.batches",
        __plum_pipeline__: ds.__plum_datasource__, 
        dataset: ds, 
        batch_size: batch_size,
        num_workers: num_workers,
        shuffle: shuffle,
        pipelines: pipelines,
        collate_funcs: collate_funcs,
        sort: sort,
        sort_key: sort_key,
        sort_descending: sort_descending,
    },

    pipeline: {
        select(fields, type=null): {
            __plum_type__: "dataio.pipeline.select",
            fields: fields,
            type: type,
        },
        batch_ndtensor(batch_dim=0): {
            __plum_type__: "dataio.pipeline.batch_ndtensor",
            batch_dim: batch_dim,
        },
        batch_sequence_ndtensor(sequence_dim, pad_value, batch_dim=0): {
            __plum_type__: "dataio.pipeline.batch_sequence_ndtensor",
            batch_dim: batch_dim,
            sequence_dim: sequence_dim,
            pad_value: pad_value,
        },
        batch_flat(): {__plum_type__: "dataio.pipeline.batch_flat",},
        vocab_lookup(vocab): {
            __plum_type__: "dataio.pipeline.vocab_lookup",
            vocab: vocab,
        },
        pad_list(pad, start=true, end=true): {
            __plum_type__: "dataio.pipeline.pad_list",
            pad: pad,
            start: start,
            end: end,
        },   
        aggregate(pipeline_ops): {
            __plum_type__: "dataio.pipeline.aggregate_list",
            fields: pipeline_ops,
        },
        pad_dim2max(dim, pad=0): {
            __plum_type__: "dataio.pipeline.pad_dim_to_max",
            pad_value: pad,
            pad_dim: dim,
        },
        cat(dim): {
            __plum_type__: "dataio.pipeline.cat",
            dim: dim,
        },
        len(): {__plum_type__: "dataio.pipeline.len"},
        long_tensor(): {__plum_type__: "dataio.pipeline.long_tensor"},
    },

};


data
