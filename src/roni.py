"""
This file contains helpers for calculating the roni tool risk score
"""

import pandas as pd
import numpy as np

def calculate_roni_scores(df, threshold=None):
    """
    Calculates a roni risk score for each student based on which risk factors they have. 
    
    Roni tool risk factors, score given if the student has that characteristic and columns from which risk factor is measured in our dataset:
    
    Attendance <90%, weighting = 1, columns = [`total_absences`]
    Attendance <85%, weighting = 2, columns = [`total_absences`]
    English as additional language, weighting = 1, columns = [`language_eng`, `language_enb`]
    ECHP, weighting = 2, columns = [`sen_provision_e`, `sen_provision_s`,`send_flag`,`sen_s`,`sen_e`]
    Special Educational Needs (SEN), weighting = 1, columns = [`sen_provision_k`,`sen_a`,`sen_k`,`sen_p`,`sen_support_flag_1`]
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
        dataframe with additional columns added for each roni tool risk factor weighting and the overall roni score for each student
    """
        
    roni_df = pd.DataFrame()
    roni_df['roni_att_90'] = np.where((df['total_absences']>0.1) & (df['total_absences']<=0.15),1,0)
    roni_df['roni_att_85'] = np.where(((df['total_absences']>0.15) & (df['total_absences']<=1.)),2,0)
    roni_df['roni_eal'] = np.where((df['language_eng']==0)&(df['language_enb']==0),1,0)
    roni_df['roni_ehcp'] = np.where((df['sen_provision_e']==1)|(df['sen_provision_s']==1)|(df['send_flag']==1)|(df['sen_s']==1)|(df['sen_e']==1),2,0)
    roni_df['roni_sen'] = np.where((df['sen_provision_k']==1)|(df['sen_a']==1)|(df['sen_k']==1)|(df['sen_p']==1)|(df['sen_support_flag_1']==1),1,0) 
    roni_df['roni_excl_20less'] = np.where((df['excluded_authorised']<=0.5) & (df['excluded_authorised']>0.),1,0) 
    roni_df['roni_excl_20more'] = np.where((df['excluded_authorised']>0.5),2,0) 
    roni_df['roni_not_sch'] = np.where((df['characteristic_code_200']==1),1,0) 
    roni_df['roni_lac'] = np.where((df['characteristic_code_110']==1), 2,0)
    roni_df['roni_preg_or_parent'] = np.where((df['characteristic_code_180']==1) | (df['characteristic_code_120']==1) | (df['characteristic_code_190']==1), 2,0)
    roni_df['roni_fsme'] = np.where((df['fsme_on_census_day_1']==1), 2,0)
    roni_df['roni_carer'] = np.where((df['characteristic_code_140']==1), 2,0)
    roni_df['roni_yot'] = np.where((df['characteristic_code_170']==1), 2,0)
    
    roni_df['roni_score'] = roni_df.sum(axis=1)
    
    if threshold is not None:
        roni_df['roni_prediction'] = (roni_df['roni_score'] >= threshold).astype(int)
    
    return roni_df
