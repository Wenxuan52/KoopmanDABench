import h5py

for fn in ["raw_data/weatherbench_train.h5", "raw_data/weatherbench_test.h5"]:
    with h5py.File(fn, "r") as f:
        print(fn, f["data"].shape, f["data"].dtype)