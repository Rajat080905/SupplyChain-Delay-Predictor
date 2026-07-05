"""
visualization.py

Creates plots for model evaluation.

Author: Rajat
"""

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


class Visualizer:

    @staticmethod
    def plot_confusion_matrix(cm, title):

        plt.figure(figsize=(6,5))

        sns.heatmap(
            cm,
            annot=True,
            cmap="Blues",
            fmt="d"
        )

        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.tight_layout()

        plt.savefig(
            f"results/{title.replace(' ','_').lower()}.png",
            dpi=300
        )

        plt.show()

    @staticmethod
    def plot_roc_curve(fpr, tpr, roc_auc, title):

        plt.figure(figsize=(6,5))

        plt.plot(
            fpr,
            tpr,
            label=f"AUC = {roc_auc:.3f}",
            linewidth=2
        )

        plt.plot(
            [0,1],
            [0,1],
            linestyle="--"
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.title(title)

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            f"results/{title.replace(' ','_').lower()}_roc.png",
            dpi=300
        )

        plt.show()
