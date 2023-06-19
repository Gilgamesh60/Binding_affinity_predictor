class Config:
    root = "/content/drive/MyDrive/refined_affinity/"
    data_dir = "/content/drive/MyDrive/refined_affinity/refined-set"
    affinity_file = "/content/drive/MyDrive/binding_affinity_project/general-set-except-refined/index/INDEX_refined_data.2016"
    batch_size = 64
    learning_rate = 0.06
    use_scheduler = False
    step_size = 5
    gamma = 0.25
    in_channels = 5
    num_gnn_layers = 2
    num_linear_layers = 1
    linear_out_channels = [5, 5]
    device = "cpu"
    num_epochs = 20
    early_stop = True
    patience = 5
