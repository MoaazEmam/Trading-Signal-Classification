import pandas as pd

from src.data.labeling import label


def _labeled(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["label"].notna()]  # type: ignore


#
# class TestLabelValues:
#     def test_buy_label_on_sharp_spike(self, labeling_buy_df):
#         result = label(labeling_buy_df.copy(), N=10, M=2)
#         first = _labeled(result).iloc[0]["label"]
#         assert first == "Buy", f"Expected , got {first}"
#
#     def test_sell_label_on_sharp_crash(self, labeling_sell_df):
#         result = label(labeling_sell_df.copy(), N=10, M=2)
#         first = _labeled(result).iloc[0]["label"]
#         assert first == "Sell", f"Expected Sell, got {first}"
#
#     def test_hold_label_when_price_flat(self, labeling_hold_df):
#         result = label(labeling_hold_df.copy(), N=10, M=2)
#         labeled = _labeled(result)
#         assert len(labeled) > 0
#         assert (labeled["label"] == "Hold").all(), \
#             f"Expected all Hold, got: {labeled['label'].value_counts().to_dict()}"
#
#     def test_only_valid_label_values(self, labeling_multi_company_df):
#         result = label(labeling_multi_company_df.copy(), N=10, M=2)
#         assert set(_labeled(result)["label"].unique()).issubset({"Buy", "Sell", "Hold"})


class TestWarmupAndTailExclusion:
    def test_warmup_rows_are_unlabeled(self, labeling_buy_df):
        """
        The labeler drops the first 20 rows per company before iterating,
        then writes labels back via original index — those rows stay NaN.
        We verify at least 20 NaN rows exist per company.
        """
        result = label(labeling_buy_df, N=10, M=2)
        company_df = result[result["Company"] == "TEST"].sort_values("Date")
        nan_rows = company_df[company_df["label"].isna()]
        assert (
            len(nan_rows) >= 20
        ), f"Expected at least 20 NaN warmup rows, got {len(nan_rows)}"

    def test_last_n_rows_per_company_are_nan(self, labeling_buy_df):
        N = 10
        result = label(labeling_buy_df, N=N, M=2)
        last_n = result[result["Company"] == "TEST"].sort_values("Date").iloc[-N:]
        assert last_n["label"].isna().all()

    def test_total_nan_count_at_least_warmup_plus_tail(self, labeling_buy_df):
        N = 10
        result = label(labeling_buy_df, N=N, M=2)
        assert result["label"].isna().sum() >= (20 + N)


class TestMultiCompany:
    # def test_companies_labeled_independently(self, labeling_multi_company_df):
    #     result  = label(labeling_multi_company_df.copy(), N=10, M=2)
    #     buy_co  = _labeled(result[result["Company"] == "BUY_CO"])
    #     sell_co = _labeled(result[result["Company"] == "SELL_CO"])
    #     assert len(buy_co) > 0 and len(sell_co) > 0
    #     assert buy_co.iloc[0]["label"]  == "Buy"
    #     assert sell_co.iloc[0]["label"] == "Sell"

    def test_label_column_exists_for_all_companies(self, labeling_multi_company_df):
        result = label(labeling_multi_company_df, N=10, M=2)
        for company in result["Company"].unique():
            assert "label" in result[result["Company"] == company].columns


class TestReturnBehaviour:
    def test_returns_dataframe(self, labeling_buy_df):
        assert isinstance(label(labeling_buy_df, N=10, M=2), pd.DataFrame)

    def test_label_column_added(self, labeling_buy_df):
        assert "label" in label(labeling_buy_df, N=10, M=2).columns

    def test_original_columns_preserved(self, labeling_buy_df):
        result = label(labeling_buy_df, N=10, M=2)
        assert set(labeling_buy_df.columns).issubset(set(result.columns))

    # def test_row_count_unchanged(self, labeling_buy_df):
    #     assert len(label(labeling_buy_df,N=10, M=2)) == len(labeling_buy_df)


class TestParameters:
    def test_large_M_produces_more_holds(self, labeling_buy_df):
        holds_tight = (
            _labeled(label(labeling_buy_df, N=10, M=0.1))["label"] == "Hold"
        ).sum()
        holds_wide = (
            _labeled(label(labeling_buy_df, N=10, M=100))["label"] == "Hold"
        ).sum()
        assert holds_wide >= holds_tight

    def test_smaller_N_leaves_fewer_tail_nans(self, labeling_buy_df):
        nan_n5 = label(labeling_buy_df, N=5, M=2)["label"].isna().sum()
        nan_n10 = label(labeling_buy_df, N=10, M=2)["label"].isna().sum()
        assert nan_n5 <= nan_n10


class TestOnSampleDf:
    def test_label_runs_without_error(self, sample_df):
        result = label(sample_df, N=10, M=2)
        assert "label" in result.columns

    def test_only_valid_label_values_in_sample(self, sample_df):
        result = label(sample_df, N=10, M=2)
        actual = set(_labeled(result)["label"].unique())
        assert actual.issubset({"Buy", "Sell", "Hold"})

    # def test_row_count_unchanged_on_sample(self, sample_df):
    #     assert len(label(sample_df,N=10, M=2)) == len(sample_df)

    # def test_nan_count_is_reasonable_on_sample(self, sample_df):
    #     """
    #     If this fails with 100% NaN, the sample was generated before
    #     labeling ran — re-run save_sample() after labeling.py.
    #     """
    #     result = label(sample_df, N=10, M=2)
    #     nan_pct = result["label"].isna().sum() / len(result)
    #     assert nan_pct < 0.50, (
    #         f"Too many NaN labels ({nan_pct:.1%}). "
    #         "Re-run save_sample() after labeling.py has been run."
    #     )

    def test_all_companies_receive_at_least_one_label(self, sample_df):
        result = label(sample_df, N=10, M=2)
        for company, group in result.groupby("Company"):
            if len(group) < 31:
                continue  # not enough rows for warmup + label + tail
            assert (
                group["label"].notna().any()
            ), f"Company {company} ({len(group)} rows) has no labeled rows"
