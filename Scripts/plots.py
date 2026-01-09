import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Correlation Heatmap


def plot_correlation_heatmap(correlation_matrix, folder_to_save, file_name):
    import os

    file_name = file_name
    folder_name = folder_to_save
    parent_dir = os.path.dirname(os.getcwd())
    destination_folder = os.path.join(parent_dir, folder_name)
    os.makedirs(destination_folder, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        square=True,
        linewidths=0.5,
        cbar={"Shrink": 0.8},
        ax=ax,
        annot_kws={"size": 8},
    )

    ax.set_title("Stocks Correlation Matrix")

    plt.tight_layout()

    if not os.path.exists(f"{destination_folder}/{file_name}"):
        plt.savefig(f"{destination_folder}/{file_name}", dpi=300, bbox_inches="tight")
        print(f"Correlation matrix image saved to {destination_folder}")
    else:
        print("An image for correlation matrix already exists and will be replaced")
    plt.show()
