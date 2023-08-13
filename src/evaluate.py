import numpy as np
import pandas as pd


def get_recall(df_evaluate, k):
    """
    df_evaluate: columns[preds: list, targets: list]
    """
    df_evaluate["n_hits"] = df_evaluate.apply(
        lambda row: len(set(row["preds"][:k]).intersection(set(row["targets"]))),
        axis=1,
    )
    df_evaluate["n_targets"] = df_evaluate["targets"].apply(len)
    recall = (df_evaluate["n_hits"] / df_evaluate["n_targets"]).mean()
    return recall


if __name__ == "__main__":
    n_users = 100
    df_evaluate = pd.DataFrame({
        "preds": [np.arange(5) for i in range(n_users)],
        "targets": [np.arange(5) + 3 for i in range(n_users)]
    })
    recall = get_recall(df_evaluate)
    print(f"recall: {recall}")  # 0.4
