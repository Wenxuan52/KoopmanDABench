import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def flatten_loss_list(loss_list, file_name=None, key_name=None):
    flat = []
    for i, item in enumerate(loss_list):
        if isinstance(item, (float, int, np.float32, np.float64)):
            flat.append(float(item))
        elif isinstance(item, (list, np.ndarray)):
            if len(item) == 0:
                if file_name and key_name:
                    print(f"[WARNING] Empty item at index {i} in {key_name} of {file_name}")
                continue
            flat.extend([float(x) for x in np.array(item).flatten()])
        else:
            if file_name and key_name:
                print(f"[WARNING] Unexpected type in {key_name} of {file_name}: {type(item)}")
            continue
    return flat


def load_all_losses(loss_dir):
    loss_files = sorted([
        os.path.join(loss_dir, f)
        for f in os.listdir(loss_dir)
        if f.startswith("training_loss_stage") and f.endswith(".pkl")
    ])

    all_train_forward = []
    all_val_forward = []
    all_train_id = []
    all_val_id = []

    for file in loss_files:
        with open(file, "rb") as f:
            data = pickle.load(f)

        train_loss = data["train_loss"]
        val_loss = data["val_loss"]

        train_forward = flatten_loss_list(train_loss["forward"], file_name=file, key_name="train_forward")
        val_forward = flatten_loss_list(val_loss["forward"], file_name=file, key_name="val_forward")
        train_id = flatten_loss_list(train_loss["id"], file_name=file, key_name="train_id")
        val_id = flatten_loss_list(val_loss["id"], file_name=file, key_name="val_id")
        all_train_forward.extend(train_forward)
        all_val_forward.extend(val_forward)
        all_train_id.extend(train_id)
        all_val_id.extend(val_id)

    all_train_forward = np.array(all_train_forward)
    all_val_forward = np.array(all_val_forward)
    all_train_id = np.array(all_train_id)
    all_val_id = np.array(all_val_id)

    train_total = all_train_forward + all_train_id
    val_total = all_val_forward + all_val_id

    return {
        "train_forward": all_train_forward,
        "val_forward": all_val_forward,
        "train_id": all_train_id,
        "val_id": all_val_id,
        "train_total": train_total,
        "val_total": val_total
    }


def plot_loss_curves(losses, save_path="figures/loss/cylinder_losses.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_epoch = int(np.argmin(losses["val_total"]))
    best_val = losses["val_total"][best_epoch]
    print(f"[INFO] Best validation epoch: {best_epoch}, val_loss = {best_val:.6f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Forward Loss ---
    axes[0].plot(losses["train_forward"], label="Train", linewidth=2)
    axes[0].plot(losses["val_forward"], label="Val", linewidth=2)
    axes[0].axvline(best_epoch, color='red', linestyle='--', linewidth=2)
    axes[0].set_title("Forward Loss", fontsize=15)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True)

    # --- Reconstruct Loss ---
    axes[1].plot(losses["train_id"], label="Train", linewidth=2)
    axes[1].plot(losses["val_id"], label="Val", linewidth=2)
    axes[1].axvline(best_epoch, color='red', linestyle='--', linewidth=2)
    axes[1].set_title("Reconstruct Loss", fontsize=15)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True)

    # --- Total Loss ---
    axes[2].plot(losses["train_total"], label="Train", linewidth=2)
    axes[2].plot(losses["val_total"], label="Val", linewidth=2)
    axes[2].axvline(best_epoch, color='red', linestyle='--', linewidth=2, label=f"Best Val at Epoch: {best_epoch}")
    axes[2].set_title("Total Loss", fontsize=15)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_yscale("log")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)


loss_dir = "best_kol_model_weights/losses"
losses = load_all_losses(loss_dir)
plot_loss_curves(losses, save_path='figures/loss/kol_losses_train.png')
