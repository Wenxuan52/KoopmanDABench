from era5_multi_DA import run_multi_da_experiment


if __name__ == "__main__":
    # run_multi_da_experiment(mode="default")

    run_multi_da_experiment(mode="custom", config_path="configs/DA/demo_config.yaml")

