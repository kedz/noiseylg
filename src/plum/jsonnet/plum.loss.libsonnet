{
    cross_entropy(labels_field, padding_index=null) : 
        {
            __plum_type__: "loss_functions.class_cross_entropy",
            labels_field: labels_field,
        } + (
            if padding_index != null then 
                {padding_index: padding_index}
            else 
                {padding_index: -1}
        )
}
