import pandas as pd
import numpy as np

from src import data_utils
from src import log_utils


def test_all_equal_basic():
    tdf = pd.DataFrame(
        {
            "Animal": ["Falcon", "Falcon", "Fox", "Cow"],
            "Max Speed": [26.0, 26.0, 26.0, 26.0],
        }
    )
    expected = pd.Series(data=[False, True], index=["Animal", "Max Speed"])
    pd.testing.assert_series_equal(data_utils.all_equal(tdf), expected)


def test_all_equal_partial_nans():
    tdf = pd.DataFrame(
        {
            "Animal": ["Falcon", "Falcon", np.nan, np.nan],
            "Max Speed": [26.0, 26.0, 26.0, 26.0],
        }
    )
    expected = pd.Series(data=[False, True], index=["Animal", "Max Speed"])
    pd.testing.assert_series_equal(data_utils.all_equal(tdf), expected)


def test_all_equal_all_nans():
    tdf = pd.DataFrame(
        {
            "Animal": [np.nan, np.nan, np.nan, np.nan],
            "Max Speed": [26.0, 26.0, 26.0, 26.0],
        }
    )
    expected = pd.Series(data=[True, True], index=["Animal", "Max Speed"])
    pd.testing.assert_series_equal(data_utils.all_equal(tdf), expected)


def test_all_equal_empty():
    tdf = pd.DataFrame({"Animal": [], "Max Speed": []})
    expected = pd.Series(data=[True, True], index=["Animal", "Max Speed"])
    pd.testing.assert_series_equal(data_utils.all_equal(tdf), expected)


def test_get_dummies_with_logging_basic():
    tdf = pd.DataFrame(
        {
            "Animal": ["Falcon", "Falcon", "Fox", "Cow"],
            "Max Speed": [26.0, 26.0, 26.0, 26.0],
        }
    )
    test_col = ["Animal"]
    test_prefix_sep = "__"
    expected = pd.DataFrame(
        {
            "Max Speed": [26.0, 26.0, 26.0, 26.0],
            "Animal__Cow": [0, 0, 0, 1],
            "Animal__Falcon": [1, 1, 0, 0],
            "Animal__Fox": [0, 0, 1, 0],
        }
    )
    # print (data_utils.get_dummies_with_logging(tdf, test_col, prefix_sep=test_prefix_sep, logger=log_utils.PrintLogger))
    # print (expected)

    pd.testing.assert_frame_equal(
        data_utils.get_dummies_with_logging(
            tdf, test_col, prefix_sep=test_prefix_sep, logger=log_utils.PrintLogger
        ),
        expected,
        check_dtype=False,
    )  # uint8 vs int64 dtypes?


def test_get_dummies_with_logging_nans():
    tdf = pd.DataFrame(
        {
            "Animal": ["Falcon", "Falcon", "Fox", np.nan],
            "Max Speed": [26.0, 26.0, 26.0, 26.0],
        }
    )
    test_col = ["Animal"]
    test_prefix_sep = "__"
    expected = pd.DataFrame(
        {
            "Max Speed": [26.0, 26.0, 26.0, 26.0],
            "Animal__Falcon": [1, 1, 0, 0],
            "Animal__Fox": [0, 0, 1, 0],
        }
    )

    pd.testing.assert_frame_equal(
        data_utils.get_dummies_with_logging(
            tdf, test_col, prefix_sep=test_prefix_sep, logger=log_utils.PrintLogger
        ),
        expected,
        check_dtype=False,
    )


def test_drop_duplicate_rows_basic():
    tdf = pd.DataFrame(
        {
            "Animal": ["Falcon", "Falcon", "Fox", "Cow"],
            "Max Speed": [26.0, 26.0, 26.0, 26.0],
        }
    )
    expected = pd.DataFrame(
        {"Animal": ["Falcon", "Fox", "Cow"], "Max Speed": [26.0, 26.0, 26.0]},
        index=[0, 2, 3],
    )  # should the indexes be reset?

    pd.testing.assert_frame_equal(
        data_utils.drop_duplicate_rows(tdf, logger=log_utils.PrintLogger), expected
    )


def test_drop_empty_cols_basic():
    tdf = pd.DataFrame(
        {
            "Animal": ["Falcon", "Falcon", "Fox", "Cow"],
            "Max Speed": [np.nan, np.nan, np.nan, np.nan],
            "Number of legs": [2, 2, 4, 4],
        }
    )
    expected = pd.DataFrame(
        {"Animal": ["Falcon", "Falcon", "Fox", "Cow"], "Number of legs": [2, 2, 4, 4]}
    )
    NA_VALS = ["", "#VALUE!"]  # must be a list

    pd.testing.assert_frame_equal(
        data_utils.drop_empty_cols(tdf, na_vals=NA_VALS, logger=log_utils.PrintLogger),
        expected,
    )


def test_drop_single_valued_cols_basic():
    tdf = pd.DataFrame(
        {
            "Animal": ["Falcon", "Falcon", "Fox", "Cow"],
            "Max Speed": [26.0, 26.0, 26.0, 26.0],
            "Number of legs": [2, 2, 4, 4],
        }
    )
    expected = pd.DataFrame(
        {"Animal": ["Falcon", "Falcon", "Fox", "Cow"], "Number of legs": [2, 2, 4, 4]}
    )
    pd.testing.assert_frame_equal(
        data_utils.drop_single_valued_cols(tdf, logger=log_utils.PrintLogger), expected
    )


def test_drop_empty_upns_basic():
    tdf = pd.DataFrame(
        {
            "Animal": [np.nan, "Falcon", "Fox", "Cow"],
            "Max Speed": [26.0, 26.0, 26.0, 26.0],
            "Number of legs": [2, 2, 4, 4],
        },
        index=[0, 1, 2, 3],
    )
    expected = pd.DataFrame(
        {
            "Animal": ["Falcon", "Fox", "Cow"],
            "Max Speed": [26.0, 26.0, 26.0],
            "Number of legs": [2, 4, 4],
        },
        index=[1, 2, 3],
    )
    upn_col = "Animal"
    NA_VALS = ["", "#VALUE!"]  # must be a list
    print(expected)
    print(
        data_utils.drop_empty_upns(
            tdf, upn_col, na_vals=NA_VALS, logger=log_utils.PrintLogger
        )
    )

    pd.testing.assert_frame_equal(
        data_utils.drop_empty_upns(
            tdf, upn_col, na_vals=NA_VALS, logger=log_utils.PrintLogger
        ),
        expected,
    )


# def test_load_csv_basic():
#    expected = pd.DataFrame()
#
#    pd.testing.assert_frame_equal(df = data_utils.load_csv(
#    fp,
#    drop_empty=False,
#    drop_single_valued=False,
#    drop_missing_upns=False,
#    drop_duplicates=False,
#    read_as_str=False,
#    use_na=False,
#    upn_col=UPN,
#    na_vals=NA_VALS,
#    convert_dtypes=False,  # This will use the experimental pd.NA. Code from pre_age15_multi_upn_categorical will try to use this. If #read_as_str=True, then all columns will be turned into strings. Otherwise, it will infer types
#    downcast=False,
#    logger=l.PrintLogger
# ),expected)

# def test_downcast_df_basic():
#    tdf = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
#                                  'Fox', 'Cow'],
#                       'Max Speed': [26., 26., 26., 26.],
#                       'Number of legs':[2,2,4,4]})
#    expected = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
#                                  'Fox', 'Cow'],
#                       'Max Speed': [26., 26., 26., 26.],
#                       'Number of legs':[2,2,4,4]})
#    pd.testing.assert_frame_equal(data_utils.downcast_df(tdf, inplace=False),expected)
