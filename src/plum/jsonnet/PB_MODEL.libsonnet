{
    load(path, model_name=null) : {
        __plum_model__: if model_name != null then model_name else path,
        __plum_model_load__: path,
    },
    model(model_name, inputs={}, networks={}, initializers={}): {
        __plum_model__: model_name,
        __plum_type__: "plum.models.generic_model",
        inputs: inputs,
        networks: networks,
        initializers: initializers,
    },
    encoder_decoder(model_name, encoder_inputs=["source_inputs"], 
                    decoder_inputs=["target_inputs"],
                    encoder=null, decoder=null, initializers={},
                    search_algos={}, controls=null, control_inputs=[]): {

        __plum_model__: model_name,
        __plum_type__: "plum.models.encoder_decoder",

        encoder_inputs: encoder_inputs,
        encoder: encoder,

        decoder_inputs: decoder_inputs,
        decoder: decoder,

        search_algos: search_algos,

        initializers: initializers,
    } + (
        if controls != null then
            {controls: controls, control_inputs: control_inputs}
        else 
            {}
    ),
}
