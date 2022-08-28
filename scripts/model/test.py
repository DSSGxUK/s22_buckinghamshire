import pandas as pd
import os
import sys
import numpy as np
import argparse
import pickle as pkl

from src import data_utils as d
from src import file_utils as f
from src import log_utils as l
from src import cv

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

#DVC filepaths
from src.params import (
    SINGLE_TEST_FP,
    MULTI_TEST_FP,
    MODEL_FP,
    TEST_RESULTS_CSV_FP,
    PREDICTIONS_CSV_FP,
    TARGET
)

#DVC params
from src.constants import (
    CensusDataColumns,
    KSDataColumns,
    AttendanceDataColumns,
    CCISDataColumns,
    UPN,
    NA_VALS
    
)

# Non DVC params but necessary to import
from src.params import (
    get_random_seed
)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true',
                    help='run transform in debug mode')
parser.add_argument('--single', action='store_true',
                    help='Use the single upn dataset')
parser.add_argument('--output_predictions', action='store_true',
                    help='Save predictions to csv file')

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    
    logger.info(f'Processing the {"single" if args.single else "multi"} upn dataset')
    
    test_data = SINGLE_TEST_FP if args.single else MULTI_TEST_FP
    
  # Load the test dataset

    df = d.load_csv(
        test_data, 
        drop_empty=True, 
        drop_single_valued=False,
        drop_duplicates=True,
        read_as_str=False,
        drop_missing_upns=True,
        use_na=True,
        upn_col=UPN,
        na_vals=NA_VALS,
        convert_dtypes=True,
        logger=logger
    )
    
    df_original = df.copy(deep=True)
    
    index_cols = [UPN] if args.single else [UPN, YEAR]
    logger.info(f'Dropping index cols {index_cols}')
    extra_cols = [CensusDataColumns.has_census_data, KSDataColumns.has_ks2_data, AttendanceDataColumns.has_attendance_data]
    logger.info(f'Dropping extra cols {extra_cols}, since they are not used for modeling')
    d.safe_drop_columns(df, index_cols + extra_cols, inplace=True)
    
    if df.isna().any().any():
        logger.error("Data has missing values. We don't know how to handle missing data yet")
        raise NotImplementedError()
    
    # Get the numpy array data
    y = df[TARGET].astype(int).values
    df.drop(TARGET, axis=1, inplace=True)
    X = df.astype(float).values
           
    if args.single: 
        
        # Load the model.pkl file
        with open(f.tmp_path(MODEL_FP), 'rb') as model_file: 
            model = pkl.load(model_file)
                    
        # Run on test data
        logger.info(f'Predicting using {model.estimator}, threshold_type: {model.threshold_type}, threshold: {model.threshold}')
        
        # Output probablities and predictions based on threshold
        if model.threshold_type == 'decision_function' :
            y_prob = model.decision_function(X) # convert to a probability between 0 and 1 before sending to csv?   
            predictions = [0 if i<=model.threshold else 1 for i in y_prob]
        elif model.threshold_type == 'predict_proba' :
            y_prob = model.predict_proba(X)[::,1]
            predictions = [0 if i<=model.threshold else 1 for i in y_prob]
        else :
            raise ValueError(f'Unknown threshold type {model.threshold_type}')
                
        #breakpoint()
        
        # Get scores
        f1 = f1_score(y, predictions, average='binary')
        fbeta = fbeta_score(y, predictions, beta=2, average='binary')
        recall = recall_score(y, predictions, average='binary')
        precision = precision_score(y, predictions, average='binary')
        mcc = matthews_corrcoef(y, predictions)
        accuracy = accuracy_score(y, predictions)
        roc_auc = roc_auc_score(y, y_prob, average='macro') #send y_prob
        
        test_results = {
                        'threshold_type': model.threshold_type,
                        'threshold': model.threshold,
                       'f1': f1,
                       'fbeta': fbeta,
                       'recall': recall,
                       'precision': precision,
                       'mcc':mcc,
                       'accuracy': accuracy,
                       'roc_auc': roc_auc}
        
        # Save test results
        
        if args.debug:
            TEST_RESULTS_CSV_FP = f.tmp_path(TEST_RESULTS_CSV_FP)

        logger.info(f'Saving test results to {TEST_RESULTS_CSV_FP}')        
        pd.DataFrame({k: [v] for k, v in test_results.items()}).to_csv(TEST_RESULTS_CSV_FP, index=False)       

        # Save predictions & probabilities
        # this should maybe be in a separate script as will come from a separate dataset?
        if args.output_predictions :

            output_predictions = df_original
            output_predictions['predictions'] = predictions
            output_predictions['probabilities'] = y_prob
            print (max(output_predictions['probabilities']),min(output_predictions['probabilities']))

            if args.debug :
                pd.DataFrame(output_predictions).to_csv(f.tmp_path(PREDICTIONS_CSV_FP,debug=args.debug), index=False)                 
            else:
                logger.info(f'Saving predictions to {PREDICTIONS_CSV_FP}')
                pd.DataFrame(output_predictions).to_csv(PREDICTIONS_CSV_FP, index=False)      
    
        #breakpoint()
    else:
        logger.error("We don't have code yet to test the multi-upn dataset")
        raise NotImplementedError()
