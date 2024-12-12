

config = {
    "f_lr": 4e-4,
    "score_lr": 5e-4,
    "hidden_dims": (256, 256, 256),
    "dropout_rate": 0.,
    #"layer_norm": False,
    "group_norm":True,
    "fourier_scale" : 16,
    "T": 80.0,
    "tau": 0.005,
    "data_std":0.5,
    "t_min":0.002,
    "time_dim": 8,
    "rho":7.0,
    "num_scales": 1000,
    "num_samples": 50,
    "loss_norm": 'l2',
    "solver":"heun",
    "dsm_target" : False,
    "sde_name" :"KVESDE",
}