

config = {
    "score_lr": 5e-4,
    "hidden_dims": (256, 256, 256),
    "dropout_rate": 0.,
    #"layer_norm": False,
    "group_norm":True,
    "fourier_scale" : 16,
    "T": 80.0,
    "data_std":0.5,
    "t_min":0.002,
    "time_dim": 8,
    "rho":7.0,
    "num_scales": 18,
    "num_samples": 50,
    "sde_name" :"KVESDE",
}