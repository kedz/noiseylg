local init = import 'PB_INIT.libsonnet';
local vocab = import 'PB_VOCAB.libsonnet';
local activations = import 'activations.libsonnet';
local models = import 'PB_MODEL.libsonnet';


{
    cnn(hidden_size, input_vocab, label_vocab, filter_widths=[1, 2, 3, 4, 5],
        features=[16, 32, 64, 128, 256], initializers={}) : 


    local conv_nets = [
        {
            __plum_type__: "layers.seq_conv_pool_1d",
            in_feats: hidden_size,
            out_feats: features[i],
            padding: std.floor(filter_widths[i] / 2),
            kernel_size: filter_widths[i],
            dropout: 0.25,
            #pooling: 'max',
            activation: 'relu',

        } for i in std.range(0, std.length(features) - 1)

    ];

    local sum(x, y) = x + y;
    local feature_size = std.foldl(sum, features, 0);
    local encoder = {
        __plum_type__: "layers.sequential",
        layers: [
            {
                __plum_type__: "layers.embedding",
                in_feats: vocab.size(input_vocab),
                out_feats: hidden_size,
                pad_index: vocab.pad_index(input_vocab),
                dropout: 0.25,
            },
            {
                __plum_type__: "layers.parallel",
                layers: conv_nets,
            },
            {
                __plum_type__: "layers.concat",
                dim: 1,
            },
            {
                __plum_type__: "layers.fully_connected",
                in_feats: feature_size,
                out_feats: hidden_size,
                activation: activations("relu"),
                dropout: 0.25,
            },
        ],
    };

    {
        __plum_type__: "plum.models.sequence_classifier",
        __plum_model__: "cnn.classifier",
        encoder: encoder,
        predictor: {
            __plum_type__: "layers.linear_predictor",
            in_feats: hidden_size,
            num_classes: vocab.size(label_vocab),
        },

        initializers: initializers,
    }

}
