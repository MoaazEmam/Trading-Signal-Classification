import pandas as pd

# Triple barrier algorithm


def load_dataset() -> pd.DataFrame:
    return pd.read_csv("data/raw/market_data_merged.csv")


def label(df, N: float = 10, M: float = 2):
    labels = pd.Series(index=df.index, dtype="object")
    for Company, group in df.groupby("Company"):
        group = group.sort_values(by=["Date"], ascending=True).copy()

        # calculate volatility
        group["daily_return"] = group["Close"].pct_change()  # get % change between close[i] and close[i-1]
        group["volatility"] = group["daily_return"].shift(1).rolling(window=20).std()  # std of past 20 daily returns
        indices = group.index

        # drop first 20 na rows and last N rows
        group = group.dropna(subset=["volatility"])
        group = group.iloc[:-N]

        closes = group["Close"].values
        volatilities = group["volatility"].values
        for i in range(len(group)):
            upper = closes[i] + volatilities[i] * closes[i] * M
            lower = closes[i] - volatilities[i] * closes[i] * M
            future_closes = closes[i + 1 : i + N + 1]

            row_label = "Hold"

            for future_price in future_closes:
                if future_price >= upper:
                    row_label = "Buy"
                    break
                elif future_price <= lower:
                    row_label = "Sell"
                    break

            # write back labels using original index
            labels[indices[i]] = row_label

    df["label"] = labels
    return df


if __name__ == "__main__":
    dataset = load_dataset()
    df = label(dataset, N=10, M=2)
    df.to_csv("data/raw/market_data_merged.csv", index=False)
    print(df["label"].value_counts())
