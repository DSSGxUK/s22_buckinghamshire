"""
This file contains helpers for calculating the roni tool risk score
"""

import pandas as pd

from src import data_utils as d


def calculate_roni_scores(df, threshold=None):
    """
    Calculates a roni risk score for each student based on which risk factors they have.

    Roni tool risk factors, score given if the student has that characteristic and columns from which risk factor is measured in our dataset:

    Attendance <90%, weighting = 1, columns = [`total_absences`]
    Attendance <85%, weighting = 2, columns = [`total_absences`]
    English as additional language, weighting = 1, columns = [`language_eng`, `language_enb`]
    ECHP, weighting = 2, columns = [d.to_categorical('sen_provision', 'e`), d.to_categorical('sen_provision', 's`),`send_flag`,`sen_s`,`sen_e`]
    Special Educational Needs (SEN), weighting = 1, columns = [d.to_categorical('sen_provision', 'k`),`sen_a`,`sen_k`,`sen_p`,`sen_support_flag_1`]
    Exclusions (<=50% absences), weighting = 1, columns = [`excluded_authorised`]
    Exclusions (>50% absences), weighting = 2, columns = [`excluded_authorised`]
    Educated at Alternative Provision, weighting = 1, columns = [`characteristic_code_200`]
    Looked-after, weighting = 2, columns = [`characteristic_code_110`]
    Pregnant or parent, weighting = 2, columns = [`characteristic_code_180`, `characteristic_code_190`, `characteristic_code_120`]
    Eligible for Free School Meals, weighting = 2, columns = [`fsme_on_census_day_1`]
    Carer, weighting = 2, columns = [`characteristic_code_140`]
    Supervised by Youth Offending Team, weighting = 2, columns = [`characteristic_code_170`]

    Weightings are summed for each student to calculate a roni risk score.

    Parameters
    ----------
    df: pd.DataFrame
        dataframe containing columns to calculate the roni score
    threshold: int
        threshold as to which a student is classified as NEET. e.g. if threshold = 3, every student with a roni score of 3 or greater will be classified as NEET high-risk

    Returns
    ----------
    pd.DataFrame
        dataframe with columns for each roni tool risk factor weighting and the overall roni score for each student
    """
    #breakpoint()
    roni_df = pd.DataFrame(index=df.index)
    roni_df["roni_att_below_90"] = ((df["total_absences"] > 0.1) & (df["total_absences"] <= 0.15)).astype(pd.Int8Dtype()) * 1
    roni_df["roni_att_below_85"] = ((df["total_absences"] > 0.15) & (df["total_absences"] <= 1.0)).astype(pd.Int8Dtype()) * 2
    roni_df["roni_eal"] = ((df[d.to_categorical("language", "eng")] == 0) & (df[d.to_categorical("language", "enb")] == 0)).astype(pd.Int8Dtype()) * 1
    roni_df["roni_ehcp"] = (
        (df[d.to_categorical("sen_provision", "e")] == 1)
        | (df[d.to_categorical("sen_provision", "s")] == 1)
        | (df[d.to_categorical("send_flag", "1")] == 1)
        | (df[d.to_categorical("sen", "s")] == 1)
        | (df[d.to_categorical("sen", "e")] == 1)
    ).astype(pd.Int8Dtype()) * 2
    roni_df["roni_sen"] = (
        (df[d.to_categorical("sen_provision", "k")] == 1)
        | (df[d.to_categorical("sen", "a")] == 1)
        | (df[d.to_categorical("sen", "k")] == 1)
        | (df[d.to_categorical("sen", "p")] == 1)
        | (df[d.to_categorical("sen_support_flag", "1")] == 1)
    ).astype(pd.Int8Dtype()) * 1
    roni_df["roni_excluded_1"] = ((df["excluded_authorised"] <= 0.5) & (df["excluded_authorised"] > 0.0)).astype(pd.Int8Dtype()) * 1
    roni_df["roni_excluded_2"] = (df["excluded_authorised"] > 0.5).astype(pd.Int8Dtype()) * 2
    
    roni_df["roni_alt_provision"] = d.empty_series(len(roni_df), index=roni_df.index)
    if d.to_categorical("characteristic_code", "200") in roni_df.columns:
        roni_df["roni_alt_provision"] = (roni_df[d.to_categorical("characteristic_code", "200")] == 1).astype(pd.Int8Dtype()) * 1

    roni_df["roni_lac"] = d.empty_series(len(roni_df), index=roni_df.index)
    if d.to_categorical("characteristic_code", "110") in roni_df.columns :
        roni_df["roni_lac"] = (df[d.to_categorical("characteristic_code", "110")] == 1).astype(pd.Int8Dtype()) * 2
    
    is_parent = d.empty_series(len(roni_df), index=roni_df.index)
    if d.to_categorical("characteristic_code", "180") in roni_df.columns :
        is_parent = (roni_df[d.to_categorical("characteristic_code", "180")] == 1) | is_parent
    if d.to_categorical("characteristic_code", "120") in roni_df.columns :
        is_parent = (roni_df[d.to_categorical("characteristic_code", "120")] == 1) | is_parent
    if d.to_categorical("characteristic_code", "190") in roni_df.columns : 
        is_parent = (roni_df[d.to_categorical("characteristic_code", "190")] == 1) | is_parent
    roni_df["roni_pregnant_or_parent"] = is_parent.astype(pd.Int8Dtype()) * 2

    roni_df["roni_fsme"] = (df[d.to_categorical("fsme_on_census_day", "1")] == 1).astype(pd.Int8Dtype()) * 2

    roni_df["roni_carer"] = d.empty_series(len(roni_df), index=roni_df.index)
    if d.to_categorical("characteristic_code", "140") in roni_df.columns :
        roni_df["roni_carer"] = (df[d.to_categorical("characteristic_code", "140")] == 1).astype(pd.Int8Dtype()) * 2
    roni_df["roni_yot"] = d.empty_series(len(roni_df), index=roni_df.index) 
    if d.to_categorical("characteristic_code", "170") in roni_df.columns :
        roni_df["roni_yot"] = (df[d.to_categorical("characteristic_code", "170")] == 1).astype(pd.Int8Dtype()) * 2

    roni_df["roni_score"] = roni_df.sum(axis=1)

    if threshold is not None:
        roni_df["roni_prediction"] = (roni_df["roni_score"] >= threshold).astype(pd.Int8Dtype())

    return roni_df
