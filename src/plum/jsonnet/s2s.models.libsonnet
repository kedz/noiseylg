local init = import 'PB_INIT.libsonnet';
local vocab = import 'PB_VOCAB.libsonnet';
local activations = import 'activations.libsonnet';
local models = import 'PB_MODEL.libsonnet';

local make_multi_embeddings(inputs, vocabs, emb_size) = 
    local embeddings = [
        {
            __plum_type__: "layers.embedding",
            out_feats: if std.isObject(emb_size) then 
                    emb_size[field] 
                else emb_size,
            in_feats: vocab.size(vocabs[field]),
            pad_index: vocab.pad_index(vocabs[field]),
            dropout: 0.25,
        } for field in inputs
    ]; 
    
    {
        __plum_type__: "layers.zip",
        modules: embeddings,
        aggregate: {__plum_type__: "layers.concat", dim: 2},
};

local make_embeddings(emb_vocab, emb_size) = {
    __plum_type__: "layers.embedding",
    out_feats: emb_size,
    in_feats: vocab.size(emb_vocab),
    pad_index: vocab.pad_index(emb_vocab),
    dropout: 0.25,
};


{
    rnn(hidden_size, source_vocabs, target_vocab, rnn_cell="gru",
        num_layers=2, emb_sizes=null, encoder_inputs=["source_inputs"],
        decoder_inputs="target_inputs", controls=null) : 
        
        local src_emb_sizes = if emb_sizes != null then 
                emb_sizes 
            else 
                hidden_size;

        local tgt_emb_sizes = if emb_sizes != null then 
                (if std.isObject(emb_sizes) then
                    emb_sizes[decoder_inputs]
                else
                    emb_sizes )
            else 
                hidden_size;

        local enc_embs = if std.objectHas(source_vocabs, "__plum_vocab__") then
            {
                __plum_type__: "layers.zip",
                modules: [make_embeddings(source_vocabs, src_emb_sizes)],
                aggregate: {__plum_type__: "layers.concat", dim: 2},
            }
            else    
                make_multi_embeddings(encoder_inputs, source_vocabs,
                                      src_emb_sizes);

        local dec_embs = make_embeddings(target_vocab, tgt_emb_sizes);

        local sum(x,y) = x + y;
        local emb_tot_dims = if std.objectHas(source_vocabs, "__plum_vocab__")
            then 
                src_emb_sizes
            else
                if std.isObject(src_emb_sizes) then
                    std.foldl(sum, 
                              [src_emb_sizes[f] for f in encoder_inputs],
                              0)
                else
                    src_emb_sizes * std.length(encoder_inputs);

        local encoder = {
            __plum_type__: "seq2seq.encoder.rnn",
            input_net: enc_embs,           
            in_feats: emb_tot_dims,
            out_feats: hidden_size,
            bidirectional: false,
            rnn_cell: rnn_cell,
            num_layers: num_layers,
            dropout: 0.25, 
            learn_init_state: true,
        };

        local sum(x,y) = x + y;
        local control_size = if controls != null then
            std.foldl(sum, 
                      [controls[field].out_feats 
                       for field in std.objectFields(controls)], 
                      0)
        else 0;
        local decoder = {
            __plum_type__: "seq2seq.decoder.rnn",
            
            input_net: dec_embs,

            in_feats: dec_embs.out_feats + control_size,
            out_feats: hidden_size,
            bidirectional: false,
            rnn_cell: rnn_cell,
            num_layers: num_layers,
            dropout: 0.25, 

            attention_net: {
                __plum_type__: "layers.attention.feed_forward",
                hidden_size: hidden_size,
                key_net: {
                    __plum_type__: "layers.fully_connected",
                    in_feats: hidden_size,
                    out_feats: hidden_size,
                    activation: activations("identity"),
                    has_bias: false,
                },
                query_net: {
                    __plum_type__: "layers.fully_connected",
                    in_feats: hidden_size,
                    out_feats: hidden_size,
                    activation: activations("identity"),
                    has_bias: false,
                },

            },

            pre_output_net: {
                __plum_type__: "layers.fully_connected",
                in_feats: 2 * hidden_size,
                out_feats: hidden_size,
                activation: activations("tanh"),
                dropout: 0.25,
            },
            
            predictor_net: {
                __plum_type__: "layers.linear_predictor",
                in_feats: hidden_size,
                num_classes: vocab.size(target_vocab),
            },  

        };

        local control_inputs = if controls != null then
            [field for field in std.objectFields(controls)]
        else
            [];
              

        local control_module = if controls != null then
            {
                __plum_type__: "layers.zip",
                modules: [controls[field] for field in control_inputs],
                aggregate: {__plum_type__: "layers.concat", dim: 2},
            }           
        else
            null;

        models.encoder_decoder(
            "s2s." + rnn_cell,
            encoder_inputs=encoder_inputs,
            decoder_inputs=decoder_inputs,
            encoder=encoder,
            decoder=decoder,
            search_algos={
                greedy: {
                    __plum_type__: "seq2seq.search.greedy",
                    max_steps: 200,
                    vocab: target_vocab,
                },
            },
            initializers={
                "weight & recurrent": init.xavier_normal(),
                "weight & embedding": init.normal(),
                "init_hidden_state": init.normal(),
                "bias & recurrent": init.constant(if rnn_cell == "gru" then -1 else 1),
            },
            control_inputs=control_inputs,
            controls=control_module,
        )
}
