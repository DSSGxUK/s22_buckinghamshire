"""
Predicts on the current student cohort and outputs a csv file with risk scores, neet predictions, features & shapley values for each student to be imported to the power bi dahsboard   

Parameters
-----------
--debug : bool     
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.    
--single : bool
    Whether to use the single upn dataset
--input : string
    The filepath of where the input data is located (this is data on students in the current cohort to predict on)
--output : string
    The filepath of where to save the csv containing the predictions and risk scores for each student 
--model_pkl : pickle
    The filepath of where the model pickle file is located
--roni_threshold : string
    The filepath of where the csv with the roni tool threshold is located
--additional_data : string
    The filepath of where the additional data is located that needs to be added back to the final output for the power bi dashboard 

Returns
-----------
csv file
    input dataframe with additional columns for `predictions`, `probabilities`, scaled probabilites and roni scores for importing into power bi dashboard
"""

import argparse
import pandas as pd
import pickle as pkl
import shap
from dataclasses import asdict

#need to remove this
from src import cv
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d
from src import roni

from src.constants import (
    CensusDataColumns,
    KSDataColumns,
    AttendanceDataColumns,
    CharacteristicsDataColumns,
    UPN,
    YEAR,
    NA_VALS,
    Targets
)

def scale_probs_sep_threshold(df,col,new_col,old_min,old_max,new_min,new_max):
    """
    Function to scale the probabilities for the power bi dashboard.
    The probabilities above and below the threshold are scaled separately.
    
    Parameters
    -----------
    df : pd.DataFrame
        the input dataframe with `probabilities` from the model prediction. 
    col : string
        column name in dataframe to scale (this should be the `probabilities` column)
    new_col : string
        new column name that will contain the scaled probabilities (usually `prob_scaling`)
    old_min : float
        the minimum value of the column to scale (usually either 0 or `pipeline.threshold`)
    old_max : float
        the maximum value of the column to scale (usually either `pipeline.threshold` or 1)
    new_min : float
        the minimum value of the new scaled column (usually either 0 or the threshold for the power bi dashboard (7))
    new_max : float
        the maximum value of the new scaled column (usually either the threshold for the power bi dashboard (7) or 10)
    
    Returns
    -----------
    pd.Dataframe
        The input dataframe with an additional column that contains the scaled probabilities
    """
    df[new_col] = (((new_max-(new_min)) * (df[col]-old_min)/(old_max-old_min)))+new_min
    return df

parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true',
                    help='run transform in debug mode')
parser.add_argument('--single', action='store_true', help='whether to use the single upn dataset')
parser.add_argument('--output', required=True,
                    help='where to output the csv file containing predictions and data for power bi dashboard')
parser.add_argument('--input', required=True,
                    help='where the input csv containing students for prediciting on is located')
parser.add_argument('--model_pkl', required=True,
                    help='where the model pkl file is located')
parser.add_argument('--roni_threshold', required=True,help='where the csv with roni tool threshold is located')
parser.add_argument('--additional_data', required=True,
                    help='where the csv with additional data is located')

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    
    logger.info(f'Processing the {"single" if args.single else "multi"} upn dataset')
    
    MODEL_FP = args.model_pkl
    
  # Load the dataset for prediction

    df = d.load_csv(
        args.input, 
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
    

    X_indices = pd.Index(df[UPN].unique(),name=UPN)
    
    index_cols = [UPN] if args.single else [UPN, YEAR]
    logger.info(f'Setting index cols {index_cols}')
    df.set_index(index_cols, inplace=True, drop=True)
    
    extra_cols = [CensusDataColumns.has_census_data, KSDataColumns.has_ks2_data, AttendanceDataColumns.has_attendance_data]
    logger.info(f'Dropping extra cols {extra_cols}, since they are not used for modeling')
    d.safe_drop_columns(df, extra_cols, inplace=True)
    
    if df.isna().any().any():
        raise NotImplementedError("Data has missing values. We don't know how to handle missing data yet")
    
    # Get the input data
    
    # y should be labels per student and X_indices should be an index of students
    #y = df[args.target].astype(int).groupby(level=UPN).max()
    #X_indices = y.index
    #breakpoint()
    
    # The data_df is used by the DataJoinerTransformerFunc to grab the requested indices from the data.
    data_df = d.safe_drop_columns(df, columns=asdict(Targets).values(), inplace=True).astype(float)
    df = None  # Clean up unused dataframes
    
    #breakpoint()

    # Load the model.pkl file
    with open(MODEL_FP, 'rb') as model_file: 
        model = pkl.load(model_file)

    logger.info(f'Predicting using {model["estimator"]}, threshold_type: {model["threshold_type"]}, threshold: {model["threshold"]}')

    # Create pipeline
    #PIPELINE_STEPS = 
    pipeline = model["estimator"]
    #breakpoint()
    #pipeline = Pipeline(PIPELINE_STEPS)
    preprocessor = cv.DataJoinerTransformerFunc(data_df)
    postprocessor = cv.identity_postprocessor if args.single else cv.AggregatorTransformerFunc()
    pipeline = cv.PandasEstimatorWrapper(pipeline, preprocessor=preprocessor, postprocessor=postprocessor, threshold=model["threshold"], threshold_type = model["threshold_type"])
    # Output probablities and predictions based on threshold

    if pipeline.threshold_type == 'decision_function' :
        y_prob = pipeline.decision_function(X_indices)    
        predictions = [0 if i<=pipeline.threshold else 1 for i in y_prob]
    elif pipeline.threshold_type == 'predict_proba' :
        y_prob = pipeline.predict_proba(X_indices)
        predictions = [0 if i<=pipeline.threshold else 1 for i in y_prob]
    else :
        raise ValueError(f'Unknown threshold type {pipeline.threshold_type}')
    
    #breakpoint()        

    # Save predictions & probabilities            
    PREDICTIONS_CSV_FP = args.output
    output_predictions = df_original
    output_predictions['predictions'] = predictions
    output_predictions['probabilities'] = y_prob
    #print (max(output_predictions['probabilities']),min(output_predictions['probabilities']))
    breakpoint()

    #this section adds a new column with the probabilities scaled between 0 and 10, with a threshold of 7, for the power bi dashboard
    logger.info(f'Scaling probabilities for power bi dashboard')
    new_threshold = 7 #for power bi dashboard
    threshold = pipeline.threshold
    neet = output_predictions[output_predictions['predictions']==1]
    eet = output_predictions[output_predictions['predictions']==0]
    eet = scale_probs_sep_threshold(eet,'probabilities','prob_scaling',0,threshold,0,new_threshold)
    neet = scale_probs_sep_threshold(neet,'probabilities','prob_scaling',threshold,1,new_threshold,10)
    output_predictions = pd.concat([eet,neet]).sort_index()
    breakpoint()

    feature_names = data_df.columns    
    # Feature importance 
    logger.info(f'Calculating feature importances')
    final_estimator = pipeline.estimator[-1]
    if hasattr(final_estimator, 'coef_'):
        model_coefs = pd.Series(pipeline.estimator[-1].coef_[0], index=feature_names)
    else:
        model_coefs = pd.Series(pipeline.estimator[-1].feature_importances_, index=feature_names)
    res = model_coefs.sort_values(ascending=False, key=abs)
    
    #breakpoint()
    # Shapley values - might not work with multi-upn
    if args.single :
        logger.info(f'Calculating shapley values')
        explainer = shap.Explainer(pipeline.estimator[-1])
        #explainer = shap.LinearExplainer(pipeline.estimator[-1], masker=shap.maskers.Impute(data=X), data=X)
        X_trans = pipeline.estimator[:-1].transform(data_df)        
        shap_values = explainer.shap_values(X_trans)[1] #[1]
        shap_names = ['shap_' + names for names in feature_names]
        shap_df = pd.DataFrame(shap_values,columns=shap_names)
        final_output = pd.concat([output_predictions,shap_df],axis=1)
    #breakpoint()

        #calculate roni tool scores
        logger.info(f'Calculating roni tool scores for each student')       
        #get roni tool threhsold
        RONI_FP = args.roni_threshold
        roni_results = pd.read_csv(RONI_FP)
        roni_threshold = roni_results['threshold'][0]
        #breakpoint()
        roni_df = roni.calculate_roni_scores(final_output, threshold=roni_threshold)  
        final_output = pd.concat([final_output,roni_df],axis=1) #add roni scores to final output

        ## add back information from dataset that was removed for modelling 

        logger.info(f'Adding data back in from previous datasets')

        #att <85%
        final_output.loc[final_output['total_absences'] > 0.15, 'att_below_85%'] = 1
        final_output.loc[final_output['total_absences'] <= 0.15, 'att_below_85%'] = 0
        final_output['att_below_85%'] = final_output['att_below_85%'].astype(int) 

        #level of need column
        final_output.loc[final_output[d.to_categorical(CharacteristicsDataColumns.level_of_need_code, '1')]==1,'level_of_need'] = 'intensive support'
        final_output.loc[final_output[d.to_categorical(CharacteristicsDataColumns.level_of_need_code, '2')]==1,'level_of_need'] = 'supported'
        final_output.loc[final_output[d.to_categorical(CharacteristicsDataColumns.level_of_need_code, '3')]==1,'level_of_need'] = 'minimum intervention'

        #here can load data for adding back in previous data.
        pre_model_df = d.load_csv(
        args.additional_data, 
        drop_empty=False, 
        drop_single_valued=False,
        drop_duplicates=True,
        read_as_str=False,
        drop_missing_upns=True,
        use_na=True,
        upn_col=UPN,
        na_vals=NA_VALS,
        convert_dtypes=True,
        logger=logger)

        final_output = final_output.merge(pre_model_df,how='left',on=UPN) #merge additional data with final output

        #final_output

        #breakpoint()

        if args.debug :
            pd.DataFrame(final_output).to_csv(f.tmp_path(PREDICTIONS_CSV_FP,debug=args.debug), index=False)                 
        else:
            logger.info(f'Saving predictions to {PREDICTIONS_CSV_FP}')
            pd.DataFrame(final_output).to_csv(PREDICTIONS_CSV_FP, index=False)      
    
    #breakpoint()
    else :
        logger.info("We don't have code yet to add additional data to the multi-upn predictions dataset")
        final_output = output_predictions

    #else:
    #    logger.error("We don't have code yet to test the multi-upn dataset")
    #    raise NotImplementedError()












