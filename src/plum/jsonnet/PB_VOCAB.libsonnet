{    
    new(vocab_name, ds, pipeline, start_token=null, stop_token=null,
         unknown_token=null, pad_token=null, top_k=null, at_least=0): {
        __plum_type__: "dataio.vocab_reader",
        __plum_vocab__: vocab_name,
        dataset: ds,
        pipeline: pipeline,
        start_token: start_token,
        stop_token: stop_token,
        unknown_token: unknown_token,
        pad_token: pad_token,
        top_k: top_k,
        at_least: at_least,
    },

    size(vocab): {
        __plum_type__: "dataio.vocab.size",
        __plum_vocab_op__: [vocab],
    },

    pad_index(vocab): {
        __plum_type__: "dataio.vocab.pad_index",
        __plum_vocab_op__: [vocab],
    },
    load(path, name=null): {
        __plum_type__: "dataio.load_vocab",
        __plum_vocab__: if name != null then name else path,
        __plum_vocab_no_cache__: true,
        path: path,
    }
}
