"""
RONI tool calculation 
"""

import argparse
import pandas as pd
import numpy as np
import os
import sys
#from pprint import pprint
#import inflection
#import re
#from tqdm import tqdm
#from datetime import datetime

from sklearn.metrics import (
    make_scorer,
    f1_score,
    fbeta_score,
    recall_score,
    precision_score,
    roc_auc_score,
    matthews_corrcoef,
    accuracy_score
)

# DVC filepaths
from src.params import (
    SINGLE_TRAIN_FP,
    SINGLE_TEST_FP,
    RONI_RESULTS
)

from src import file_utils as f
from src import log_utils as l
from src import data_utils as d
from src import roni

parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true',
                    help='run transform in debug mode')


def calculate_roni_scores(df) :
    #RONI tool columns in dataset
    #total_absences = % of absences (0-1)    
    #language_eng & language_enb = english  
    #looked after == characteristic_code_110
    #fsme_on_census_day_1
    #excluded_authorised = %absences that were exclusions (0-1)
    #ehcp == sen_provision_e or sen_provsion_s or send_flag or sen_s or sen_e
    #sen == sen_provision_k or sen_a or sen_k or sen_p or sen_support_flag_1
    #parent or pregnant = characteristic_code_120 or characteristic_code_190 or characteristic_code_180
    #carer = characteristic_code_140
    #sup by yot = characteristic_code_170
    #alt_provision = characteristic_code_200 (for educated not at school)
    
    ##Calculate roni scores
    
    df['roni_att_90'] = np.where((df['total_absences']>0.1) & (df['total_absences']<=0.15),1,0)
    df['roni_att_85'] = np.where(((df['total_absences']>0.15) & (df['total_absences']<=1.)),2,0)
    df['roni_eal'] = np.where((df['language_eng']==0)&(df['language_enb']==0),1,0)
    df['roni_ehcp'] = np.where((df['sen_provision_e']==1)|(df['sen_provision_s']==1)|(df['send_flag']==1)|(df['sen_s']==1)|(df['sen_e']==1),2,0)
    df['roni_sen'] = np.where((df['sen_provision_k']==1)|(df['sen_a']==1)|(df['sen_k']==1)|(df['sen_p']==1)|(df['sen_support_flag_1']==1),1,0) 
    df['roni_excl_20less'] = np.where((df['excluded_authorised']<=0.5) & (df['excluded_authorised']>0.),1,0) 
    df['roni_excl_20more'] = np.where((df['excluded_authorised']>0.5),2,0) 
    df['roni_not_sch'] = np.where((df['characteristic_code_200']==1),1,0) 
    #df['roni_not_sch']= np.where((df['EducatedAtHome']==1) | (df['NotRegistered']==1) | (df['CurrentSituationNotKnown']==1),1,0)
    df['roni_lac'] = np.where((df['characteristic_code_110']==1), 2,0)
    df['roni_preg_or_parent'] = np.where((df['characteristic_code_180']==1) | (df['characteristic_code_120']==1) | (df['characteristic_code_190']==1), 2,0)
    df['roni_fsme'] = np.where((df['fsme_on_census_day_1']==1), 2,0)
    df['roni_carer'] = np.where((df['characteristic_code_140']==1), 2,0)
    #df['roni_custody'] = np.where((df['Custody']==1) | (df['FTeduCustodialInst']==1), 2,0)
    df['roni_yot'] = np.where((df['characteristic_code_170']==1), 2,0)
    cols = ['roni_att_90','roni_att_85','roni_eal','roni_ehcp', 'roni_sen','roni_excl_20less','roni_excl_20more','roni_not_sch',
           'roni_lac','roni_preg_or_parent','roni_fsme','roni_carer','roni_yot']
    
    df['roni_score'] = df[cols].sum(axis=1)
    
    #print (df['roni_score'].value_counts()) 

    return df

if __name__ == '__main__':
    args = parser.parse_args()

    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    logger.info(f'Running roni tool calculation')
    data_fp = SINGLE_TRAIN_FP
    
    df = d.load_csv(
        data_fp, 
        drop_empty=True, 
        drop_single_valued=False,
        drop_duplicates=True,
        read_as_str=False,
        drop_missing_upns=True,
        use_na=True,
        #upn_col=UPN,
        #na_vals=NA_VALS,
        convert_dtypes=True,
        logger=logger
    )
    
    df = df.copy()
    
    if df.isna().any().any():
        logger.error("Data has missing values. We don't know how to handle missing data yet")
        raise NotImplementedError()

    roni_df = roni.calculate_roni_scores(df)
        
    thresholds = range(1,max(roni_df['roni_score'])+1)
    print (thresholds)
    fbetas = []

    fscores = []
    for th in thresholds :
        roni_df['y_pred_'+str(th)] = np.where((roni_df['roni_score']>=th),1,0)
        #breakpoint()

        print ('Threshold', th)
        print ('recall_score :',recall_score(df['neet_ever'].astype('int64'),roni_df['y_pred_'+str(th)]))
        print ('precision_score',         precision_score(df['neet_ever'].astype('int64'),roni_df['y_pred_'+str(th)]))
        print ('f1_score',         f1_score(df['neet_ever'].astype('int64'),roni_df['y_pred_'+str(th)]))
        print ('fbeta_score',         fbeta_score(df['neet_ever'].astype('int64'),roni_df['y_pred_'+str(th)],beta=2))

        fscores.append(f1_score(df['neet_ever'].astype('int64'),roni_df['y_pred_'+str(th)]))
        fbetas.append(fbeta_score(df['neet_ever'].astype('int64'),roni_df['y_pred_'+str(th)],beta=2))

    print ('Best Fbeta threshold:', np.argmax(fbetas)+1)    
    print ('Best Fscore threshold:', np.argmax(fscores)+1) 

    best_threshold = np.argmax(fbetas)+1

    '''Run RONI tool calculation on test dataset'''
    
    data_fp = SINGLE_TEST_FP
    
    df = d.load_csv(
        data_fp, 
        drop_empty=True, 
        drop_single_valued=False,
        drop_duplicates=True,
        read_as_str=False,
        drop_missing_upns=True,
        use_na=True,
        #upn_col=UPN,
        #na_vals=NA_VALS,
        convert_dtypes=True,
        logger=logger
    )
    
    df = df.copy()
    
    if df.isna().any().any():
        logger.error("Data has missing values. We don't know how to handle missing data yet")
        raise NotImplementedError()
    
    roni_df = roni.calculate_roni_scores(df, threshold=best_threshold)  
    
    recall = recall_score(df['neet_ever'].astype('int64'),roni_df['roni_prediction'])
    precision = precision_score(df['neet_ever'].astype('int64'),roni_df['roni_prediction'])
    f1 = f1_score(df['neet_ever'].astype('int64'),roni_df['roni_prediction'])
    fbeta = fbeta_score(df['neet_ever'].astype('int64'),roni_df['roni_prediction'],beta=2) 
    
    print ('Threshold:', best_threshold)
    print ('Test scores:')
    print ('recall_score :',recall)
    print ('precision_score', precision)
    print ('f1_score',  f1)
    print ('fbeta_score', fbeta)    
    
    scores = [best_threshold,recall,precision,f1,fbeta]
    
    scores_df = pd.DataFrame([np.transpose(scores)],columns=['threshold','recall','precision','f1','fbeta'])
    
    scores_df.to_csv(RONI_RESULTS,index=False)
    
    #breakpoint()
        
        
        
        

        
        
        


