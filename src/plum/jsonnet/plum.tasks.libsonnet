{
    predict(batches, output_fields, reference_fields=null, name="predict") : {
        __plum_type__: "tasks.predict",
        __plum_program__: name,
        batches: batches,
        output_fields: output_fields,

    } + if reference_fields != null then {reference_fields: reference_fields}
        else {},

}
