{
    
    normal(mean=0.0, std=1.0): {
        __plum_type__: "initializers.normal",
        mean: mean,
        std: std,
    },

    xavier_normal(): {
        __plum_type__: "initializers.xavier_normal"
    },

    constant(value): {
        __plum_type__: "initializers.constant",
        value: value,
    },

}
