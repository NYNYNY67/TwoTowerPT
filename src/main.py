import pathlib
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets import get_user_item_loader, get_user_loader
from evaluate import get_recall


class TwoTowerRetrieval:
    def __init__(
            self,
            df_data,
            lr=0.1,
            embed_dim=32,
            batch_size=8192,
            epochs=3
    ):
        self.df_data = df_data[["user_id", "item_id"]] - 1
        self.lr = lr
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.epochs = epochs

        self.unique_items = self.df_data["item_id"].unique().tolist()
        self.unique_users = self.df_data["user_id"].unique().tolist()

        print(f"n_users: {len(self.unique_users)}, n_items: {len(self.unique_items)}")

        self.df_train, self.df_valid = train_test_split(
            self.df_data,
            test_size=0.2,
            shuffle=True,
            random_state=42
        )

        self.train_users = self.df_train["user_id"].unique()
        self.valid_users = self.df_valid["user_id"].unique()

        self.items = torch.LongTensor(self.df_data["item_id"].unique())

        self.train_loader = get_user_item_loader(
            self.df_train["user_id"].values,
            self.df_train["item_id"].values,
        )
        self.valid_loader = get_user_item_loader(
            self.df_valid["user_id"].values,
            self.df_valid["item_id"].values,
        )

        self.train_user_loader = get_user_loader(
            self.train_users,
        )
        self.valid_user_loader = get_user_loader(
            self.valid_users,
        )

        self.model = MovielensModel(
            embed_dim=embed_dim,
            n_users=len(self.unique_users),
            n_items=len(self.unique_items),
        )

        self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=lr)

    def main(self):
        self.train()
        self.evaluate()

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            train_loss = self.train_epoch()
            valid_loss = self.valid_epoch()
            print(f"epoch: {epoch+1}, train loss: {train_loss}, valid loss: {valid_loss}")

    def train_epoch(self):
        self.model.train()
        train_loss = 0
        for users, items in self.train_loader:
            out = self.model(users, self.items)
            loss = self.loss_fn(out, items)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        train_loss /= len(self.train_loader)
        return train_loss

    def valid_epoch(self):
        self.model.eval()
        valid_loss = 0
        for users, items in self.valid_loader:
            out = self.model(users, self.items)
            loss = self.loss_fn(out, items)
            valid_loss += loss.item()
        valid_loss /= len(self.valid_loader)
        return valid_loss

    def evaluate(
            self,
            k=100
    ):
        print("evaluating on training data ...")
        df_train_preds = self.predict_topk(self.train_user_loader, k)
        df_train_true = (
            self.df_train
            .groupby("user_id")["item_id"]
            .apply(list)
            .reset_index()
            .rename(columns={"item_id": "targets"})
        )
        df_train_eval = df_train_preds.merge(df_train_true, on="user_id", validate="one_to_one")
        train_recall = get_recall(df_train_eval, k)

        print("evaluating on validation data ...")
        df_valid_preds = self.predict_topk(self.valid_user_loader)
        df_valid_true = (
            self.df_valid
            .groupby("user_id")["item_id"]
            .apply(list)
            .reset_index()
            .rename(columns={"item_id": "targets"})
        )
        df_valid_eval = df_valid_preds.merge(df_valid_true, on="user_id", validate="one_to_one")
        valid_recall = get_recall(df_valid_eval, k)

        print(f"training data recall@{k}: {train_recall}, validation data recall@{k}: {valid_recall}")

    def predict_topk(self, user_loader, k=100):
        pred_users = []
        preds = []
        for users in user_loader:
            out = self.model(users, self.items)  # (n_users, n_items)
            pred_users.append(users)
            preds.append(out.argsort(dim=-1, descending=True)[:, :k])
        users = torch.cat(pred_users, dim=0).tolist()
        preds = torch.cat(preds, dim=0).tolist()
        return pd.DataFrame({"user_id": users, "preds": preds})


class MovielensModel(nn.Module):
    def __init__(
            self,
            embed_dim,
            n_users,
            n_items,
    ):
        super().__init__()
        self.user_model = nn.Embedding(n_users, embed_dim)
        self.item_model = nn.Embedding(n_items, embed_dim)

    def forward(self, user_tokens, item_tokens):
        x_user = self.user_model(user_tokens)  # (n_users, embed_dim)
        x_item = self.item_model(item_tokens)  # (n_items, embed_dim)
        return torch.matmul(x_user, x_item.T)  # (n_users, n_items)


if __name__ == "__main__":
    data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"
    df_data = pd.read_csv(data_dir / "ml-100k" / "u.data", delimiter="\t", header=None)
    df_data.columns = ["user_id", "item_id", "rating", "timestamp"]
    ttr = TwoTowerRetrieval(df_data)
    ttr.main()
