local vocablib = import 'PB_VOCAB.libsonnet';

{
    s2s_eval_script(path, search_fields, references_fields): {
        __plum_type__: "metrics.seq2seq_eval_script",
        path: path,
        search_fields: search_fields,
        references_fields: references_fields,
    },

    dict(metrics): {
        __plum_type__: "metrics.metric_dict",
        metrics: metrics,
    },

    class_prf(output_field=null, targets_field=null, num_classes=null,
              vocab=null): {
        __plum_type__: "metrics.class_prf",
    } + ( 
        if output_field != null then {output_field: output_field} else {}
    ) + (
        if targets_field != null then {targets_field: targets_field} else {}
    ) + (
        if num_classes != null then 
            {num_classes: num_classes, vocab: vocab}
        else
            {num_classes: vocablib.size(vocab), vocab: vocab}
    ),
}
