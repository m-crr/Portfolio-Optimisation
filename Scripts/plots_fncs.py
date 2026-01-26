# =========================================================================================================

# Required modules

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# =========================================================================================================


# *********************************************************************************************************

# Notebook: data_collection.ipynb

# *********************************************************************************************************


# =========================================================================================================


# Correlation Heatmap


def plot_correlation_heatmap(correlation_matrix, folder_to_save, file_name):
    file_name = file_name
    folder_name = folder_to_save
    parent_dir = os.path.dirname(os.getcwd())
    destination_folder = os.path.join(parent_dir, folder_name)
    os.makedirs(destination_folder, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
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

    print("Correlation Heatmap - \n")
    plt.show()

    if not os.path.exists(f"{destination_folder}/{file_name}"):
        plt.savefig(f"{destination_folder}/{file_name}", dpi=300, bbox_inches="tight")
        print(f"Correlation matrix image saved to {destination_folder}")
    else:
        print(
            "Warning: An image of the correlation matrix heatmap already exists and will be replaced \n"
        )


# =========================================================================================================


# Rolling Volitility


def plot_rolling_vol(returns, rolling_window, stocks_list, folder_to_save, file_name):
    import os

    file_name = file_name
    folder_name = folder_to_save
    parent_dir = os.path.dirname(os.getcwd())
    destination_folder = os.path.join(parent_dir, folder_name)
    os.makedirs(destination_folder, exist_ok=True)

    rolling_vol = (returns.rolling(window=rolling_window).std()) * (np.sqrt(252))

    stocks = stocks_list
    # selected_stocks = [s for s in stocks if s in rolling_vol.columns]

    fig, ax = plt.subplots(figsize=(14, 8))
    for stock in stocks:
        sns.lineplot(
            data=rolling_vol[stock],
            ax=ax,
            palette="virdis",
            linewidth=2,
            alpha=0.7,
            label=stock,
        )
    ax.axvspan(
        "2020-02-20", "2020-05-13", alpha=0.2, color="red", label="COVID-19 Crash"
    )

    ax.legend(loc="upper right")
    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("Annualized Volatility", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Rolling {rolling_window} Day Volatility (Annualized)",
        fontsize=14,
        fontweight="bold",
    )

    if not os.path.exists(f"{destination_folder}/{file_name}"):
        plt.savefig(f"{destination_folder}/{file_name}", dpi=300, bbox_inches="tight")
        print(f"Rolling volatility plot image saved to {destination_folder} \n")
    else:
        print(
            "Warning: An image for Rolling volatility plot already exists and will be replaced \n"
        )
    plt.tight_layout()
    plt.show()


# =========================================================================================================
