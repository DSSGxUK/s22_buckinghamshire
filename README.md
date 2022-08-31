# Predicting students at risk of becoming NEET (Not in Education, Employment or Training)

Welcome to the code repository for the project conducted under **Data Science for Social Good- UK 2022 (DSSGx UK)**, for our partner: **Buckinghamshire Council**. The aim of the project was to identify students, before Year 11, at risk of becoming NEET after they complete their GCSEs. This readme will focus on documenting:

1.	Folder structure 
2.  Assumptions
3.	Setting up a machine for running all the workflows
4.	How to run different workflows
5.	Expected data schema for Power BI dashboard

# Folder Structure

```bash

s22_buckinghamshire
├── .dvc
│   ├── .gitignore                                         
│   └── .config                              #links to the dagshub repo
├── communications
│   ├── upn-different-activity-codes.md            
│   ├── upns_with_questions.md                                           
├── data		                                 
│   └── interim	                             #stores canonicalised, annotated and merged csv files
│   │   ├── attendance_canonicalized_csv
│   │   │   ├── .gitignore
│   │   ├── ccis_canonicalized_csv
│   │   │   ├── .gitignore
│   │   ├── census_canonicalized_csv
│   │   │   ├── .gitignore
│   │   ├── ks4_canonicalized_csv
│   │   │   ├── .gitignore
|   |   └── .gitignore
|   └── processed
|   |   └── .gitignore
|   └── raw                                  # stores the original files
│   │   ├── attendance_original_csv
│   │   │   ├── .gitignore
│   │   ├── ccis_original_csv
│   │   │   ├── .gitignore
│   │   ├── census_original_csv
│   │   │   ├── .gitignore
│   │   ├── ks4_original_csv
│   │   │   ├── .gitignore
│   │   ├── .gitignore
|   └── .gitignore
├── logs
│   └── .gitignore
├── metrics                     # contains metrics and results related values
│   ├──lgbm1_single.csv         # We've kept this one preloaded
│   ├──lgbm2_single.csv         # Gets created when you run the hyperparam search
├── models
│   ├── final                           # final model for prediction
│   │   ├── model_single.pkl            # Gets created when you 
│   └── interim
├── notebooks                           
│   ├── convert_synthetic.ipynb                                         
│   └── view_csv.ipynb
├── plots                              #stores different plots and charts
│   ├──attendance_percent_box_plot.png
│   ├──common_neet_traces.png
│   ├──consequent_antecedent.png
│   ├──lgb1_feature_importance.png
│   ├──neet_cons_ante.png
│   ├──neet_infrequent_traces.png
│   ├──neet_process_map_97.html
│   ├──not_known_common_traces.png
│   ├──not_known_proess_map_98.html
│   ├──notknown_consequent_antecedent.png
│   ├──pipline.dot
│   ├──pipline.dot.svg
│   ├──pipline.md
│   ├──pipline.png
│   ├──process_map.html
│   ├──roni_catch_model_miss_feature_importance.png
│   ├──sankey.png
│   ├──sankey_debug.png
│   ├──sankey_eet.png
│   ├──sankey_full.png
│   ├──sankey_neet.png
│   ├──sankey_unknown.png                                     
│   └── unknown_infrequent_traces.png   
├── results
│   ├── .gitignore                                         
├── scripts                       #python code files for different purposes
│   ├── data    
│   │   ├── additional_data.py
│   │   ├──annotate_attendance_data.py
│   │   ├──annotate_census_data.py
│   │   ├──annotate_ks4_data.py
│   │   ├──annotate_neet_data.py
│   │   ├──attendance_premerge.py
│   │   ├──build_ccis_ks_eda_dataset.py
│   │   ├──canonicalize_data.py
│   │   ├──census_premerge.py
│   │   ├──feature_selection.py
│   │   ├──feature_selection_for_predictions.py
│   │   ├──ks2_filter.py
│   │   ├──merge_data.py
│   │   ├──merge_multi_upn.py
│   │   ├──multi_upn_categorical.py
│   │   ├──multiple_to_single.py
│   │   ├──neet_premerge.py
│   │   ├──split_covid_years.py
│   |   └── xl_to_csv.py
│   ├── misc  
│   │   ├── .Rhistory
│   │   ├── bupar-analysis.r
│   │   ├── compute_intersections.py
│   │   ├── plot_sankey.py
│   ├── model
│   │   ├── optimization_and_cv.py
│   │   ├── predict.py
│   │   ├── retrain.py
│   │   ├── roni_tool.py
│   │   ├── split_data.py
│   │   ├── test.py
│   ├── dvc.lock
│   ├── dvc.yaml
│   ├── generate_params.py
│   ├── params.yaml
├── src
│   ├── constants
│   │   ├── __init__.py
│   │   ├── attendance_constants.py
│   │   ├── ccis_constants.py
│   │   ├── census_constants.py
│   │   ├── ks_constants.py
│   │   ├── modeling_constants.py
│   │   ├── school_info_constants.py
│   │   ├── script_argument_constants.py
│   │   ├── shared_constants.py
│   ├── cv
│   │   ├── __init__.py
│   │   ├── cross_validator.py
│   │   ├── processing.py
│   │   ├── search_spaces.py
│   │   ├── utils.py
│   ├── params
│   │   ├── __init__.py
│   │   ├── data_pipeline_arguments.py
│   │   ├── data_pipeline_params.py
│   │   ├── filepaths.py
│   │   ├── model_pipeline_arguments.py
│   │   ├── model_pipeline_params.py
│   ├── aggregation_utils.py
│   ├── attendance_utils.py
│   ├── ccis_utils.py
│   ├── data_utils.py
│   ├── debug_utils.py
│   ├── error_utils.py
│   ├── file_utils.py
│   ├── log_utils.py
│   ├── merge_utils.py
│   ├── py_utils.py
│   ├── roni.py
├── tests
│   ├── scripts/data
│   │   ├── test_annotate_neet_data.py
│   │   ├── test_split_data.py
│   ├── src
│   │   ├── test_cv_utils.py
│   │   ├── test_data_utils.py
│   │   ├── test_file_utils.py
│   └── .gitignore
├── .dvcignore
├── .gitignore
├── README.md
├── requirements.R
├── requirements.txt
├── setup.py

```

## Brief folder description

`data` : This folder contains two sub-folders : `interim` and `raw`. After running the pipeline an additional `processed` folder will also be present. The original dataset files are stored in their dataset sub-folder within the `raw` folder e.g. `raw/attendance_original_csv` will contain the original csv files for attendance datasets. These original files will go through the data pipeline and will generate additional files which will be canonicalized (standardised formatting), annotated and merged across years, which will be stored in `interim` sub-folder. The `processed` subfolder will contain the final datasets ready to be used for modeling.
  
`metrics` : This folder contains outputs from the hyperparameter search, roni tool performance results and our model performance results on the test dataset.

`models` : This folder contains pickle files of the models. There are two sub-folders: `interim` and `final`. `interim` holds the checkpoints. You can find more details about these in the [Reloading the hyperparameter search from a checkpoint](#reloading-the-hyperparameter-search-from-a-checkpoint) section. The final, retrained best model can be found in `models/final/model_single.pkl`.

`results` : After running the pipeline, this folder will contain the final output CSV files: `predictions.csv`, `unknown_predictions.csv`, `unidentified_students_single.csv`, `unidentified_unknowns_single.csv`. These files are outlined in more detail below under [Outputs from running the pipeline](#outputs-from-running-the-pipeline)

`scripts` : This folder contains the `dvc.yaml` file that outlines the different stages and steps of the pipeline. It also includes two main sub-folders: `data` and `model`. The `data` sub-folder contains python scripts that prepare interim and final datasets for modeling. The `model` sub-folder contains scripts that split the final dataset into train and test datasets, runs the cross-validation and hyperparameter search, re-trains the model, calculates roni scores and uses the trained model to generate predictions for current/unknown students.        

`src` : This folder contains helper functions (found in the `*_utils.py` scripts) and also contains scripts that can set different parameters. There are three sub-folders: `constants`, `cv` and `params`. The `cv` sub-folder contains helper functions for the cross-validation and hyper-parameter search stage. It also contains dictionaries of the hyper-parameter search spaces in `search_spaces.py`. The `constants` folder contains parameters for the pipeline that are unlikely to need to change, whereas the `params` sub-folder contains parameters for the pipeline that may need/want to be changed. The `*_arguments.py` scripts in this sub-folder include the arguments that are sent to the `dvc.yaml` pipeline.    

# Assumptions

Assuming that the data provided by the user are of the following types:
  1. Attendance [Attendance_Schema](https://www.gov.uk/guidance/complete-the-school-census/data-items-2022-to-2023)
  2. Census [Census_Schema](https://www.gov.uk/guidance/complete-the-school-census/data-items-2022-to-2023)
  3. CCIS [CCIS_Schema](https://www.gov.uk/government/publications/nccis-management-information-requirement)
  4. KS4 [KS4_Schema](https://explore-education-statistics.service.gov.uk/find-statistics/key-stage-4-destination-measures/2019-20#releaseHeadlines-charts)

In addition, we want to allow data on *characteristics* and *ks2*. This has not been supported yet, but would fill in features we are passing to the model for training.

**Points to remember**:
  1. Please ensure files are in CSV format only
  2. Currently columns are renamed to `snake_case` (lowercase with spaces as _). You may need to add more columns to the renaming dictionary if your columns have changed or are different. 
     You can find the renaming dictionary in the `src` directory in the `[TYPE]_utils.py` file where `[TYPE]` refers to whatever your data type is. Note *ks2* and
     *characteristics* will not currently show up in there.
  3. We assume that the CCIS datasets have a `month/year of birth` column with numeric month and year values in the form `[MONTH]/[YEAR]`. We don't use the date of
     birth column. You can safely remove it if there is concern about data sensitivity.
  4. No column names can include "__"! This is a special reserved character for our code.


# Setting up a machine for running all the workflows

This part will change slightly depending on what operating system you are using.

## Windows

1. Ensure you have an updated python installed on your machine. You can install it through the [Microsoft Store](https://www.microsoft.com/store/productId/9PJPW5LDXLZ5). As of writing this, the most up to date version was python 3.10.
2. Ensure you have git installed. You can get an installer [here](https://git-scm.com/download/win).
3. Open a powershell (as administrator if possible) and run the below steps:

```bash
> cd [DESIRED_PARENT_DIRECTORY]                                       # Navigate to the desired parent directory using
                                                                      # this (change directory) command. Fill in
                                                                      # [DESIRED_PARENT_DIRECTORY] with your desired 
                                                                      # directory.

> ls                                                                  # You can run this command to see the contents 
                                                                      # of a directory 

> git clone https://github.com/DSSGxUK/s22_buckinghamshire.git        # This will download the code repository to the
                                                                      # current folder

> cd s22_buckinghamshire                                              # This navigates to the repository folder

> python -m venv venv                                                 # Creating a python virtual environment

> .\venv\Scripts\activate                                             # Running the virtual environment. 
                                                                      # If you get an error that says '... 
                                                                      # s22_buckinghamshire\venv\Scripts\Activate.ps1'
                                                                      # cannot be loaded because running scripts is 
                                                                      # disabled on this system. For more information,
                                                                      # see about_Execution_Policies at 'https:/go.
                                                                      # microsoft.com/fwlink/?LinkID=135170',
                                                                      # then we need to enable execution of signed 
                                                                      # scripts. 
                                                                      # We can do this by running 'Set-ExecutionPolicy 
                                                                      # RemoteSigned -Scope CurrentUser'. You may need 
                                                                      # administrator privelages to do this.
                                                              
> python.exe -m pip install --upgrade pip                             # Update pip if necessary

> python.exe -m pip install -r .\requirements.txt                     # Install required python packages

```

## Mac or Linux

1. Ensure you have an updated python installed on your machine. For Mac, you can find the python installer at [python.org](https://www.python.org/downloads/macos/). For Linux, you can find the installer [here](https://www.python.org/downloads/source/)
2. Ensure you have git installed. For Mac, ou can get an installer [here](https://git-scm.com/download/mac). For Linux, you can find directions for installing [here](https://git-scm.com/download/linux)
3. Open a terminal and run the below steps:

```bash
> cd [DESIRED_PARENT_DIRECTORY]                                       # Navigate to the desired parent directory using
                                                                      # this (change directory) command. Fill in
                                                                      # [DESIRED_PARENT_DIRECTORY] with your desired 
                                                                      # directory.

> ls                                                                  # You can run this command to see the contents 
                                                                      # of a directory

> git clone https://github.com/DSSGxUK/s22_buckinghamshire.git        # This will download the code repository to the
                                                                      # current folder

> cd s22_buckinghamshire                                              # This navigates to the repository folder

> python -m venv venv                                                 # Creating a python virtual environment

> source .\venv\Scripts\activate                                      # Running the virtual environment. 
                                                              
> pip install --upgrade pip                                           # Update pip if necessary

> pip install -r .\requirements.txt                                   # Install required python packages

```

# How to run different workflows

## Connecting Data

### Downloading the synthetic data

We've published synthetic data to dagshub so you can play around with the pipeline. This data is randomly generated and any unique IDs are random strings.

 To retrieve it, please run the following
```bash
dvc remote add origin https://dagshub.com/abhmul/s22_buckinghamshire.dvc
dvc pull -r origin
```

### Using your own data

If you are a council with your own data, these datasets will need to be saved in the `data/raw` directory as csv files in the correct formats with the correct column names. 

@to be done? We have examples of what this should look like here...

Within the `data/raw` directory are 4 folders that correspond to the different datasets listed above under *Assumptions*: 
- `attendance_original_csv`
- `ccis_original_csv`
- `census_original_csv`
- `ks4_original_csv`

The datasets in these directories should be named `[TYPE]_original_[DATE].csv` where `[TYPE]` refers to the dataset (attendance, ccis, census, ks4) and `[DATE]` refers to the month and year the dataset was submitted (e.g. `attendance_original_jan21.csv` corresponds to autumn 2021 attendance data, which is submitted in January). `[DATE]` should be written as the first 3 letters of the month and the last 2 digits of the year e.g. `jan21`, `sep19`.

### Adding New Columns

We currently do not support addition of new columns. The code should work fine if you add new columns but it will not use them in modeling.

## Running the code

**Please follow the below steps before running the workflows**:
  
```bash
cd .\scripts\
```

### Run the whole pipeline

To run the whole pipeline you can run:

```bash
dvc repro
```
(This will include a hyper parameter search which can take a few hours to run)

Alternatively, you could run the steps individually:

```bash
  # Generate datasets for modelling
  dvc repro -s --glob generate_modeling_*

  # Run cross validation and hyper parameter search 
  dvc repro -s --glob cv_*

  # Model Evaluation 
  dvc repro -s --glob model_evaluation_* 

  # Generate datasets for predictions and final output 
  dvc repro -s --glob prediction_* 
```

  ### Output predictions on new data without re-running the hyper parameter search  
  
  Following these steps re-trains the model with new data using the previous best hyper parameters.
    
```bash
  # Generate datasets for modelling
  dvc repro -s --glob generate_modeling_*          

  # Model Evaluation 
  dvc repro -s --glob model_evaluation_*                        

  # Generate datasets for predictions and final output
  dvc repro --glob prediction_* 
```

Below is a brief overview of what each stage within a workflow is doing:

**Generate datasets for modelling**
  - Merges each dataset (eg: Census, Attendance, etc) across all the years 
  - Split categorical variables into binary columns containing 0's or 1's 
  - Drop columns which aren't required for modelling or for which we won't have data available before Year 11
  - Output two datasets ready for modelling: 
    - Only unique students (UPNs)
    - A student having multiple observations across different years
    
**Run cross validation and hyper parameter search**
  - Searches for the best model parameters. Please note you can opt out of running this step. 
  - This search includes checkpoints. To rerun the hyperparameter search from a checkpoint:
    - Change the `LOAD_CHECKPOINTS` value in `scripts/params.yaml` to `True`.
  - When re-running the search with new data, ensure the `LOAD_CHECKPOINTS` value is set to `False` (otherwise an old checkpoint will be used for the new data).   
    
**Model Evaluation**
  - Evaluates RONI tool's performance
  - Retrains model with new incoming data and saves model with best threshold
  - Apply the chosen model on the test data to output a final model performance score
  
**Generate datasets for predictions and final output**
  - Creates datasets with current Year 7-10 students and students with unknown destinations to predict on
  - Executes model on unseen data and generates final predictions in form of a CSV
  - Generates feature importance
  - Returns RONI score
  - Returns scaled probability scores for a student at risk of becoming NEET (between 1-10)

### Reloading the hyperparameter search from a checkpoint

Because the hyperparameter search takes a long time, we have built support for checkpoint progress. If for some reason the run does not complete, you can pick it up from where it left off rather than restarting it. To do this you will need to complete the following steps:
1. Open `src/params/model_pipeline_params.py`.
2. Find the variable `LOAD_CHECKPOINTS`. Change its value to `True`.
3. From the `scripts` folder (you may already be there if you were running the pipeline), run `python generate_params.py`. This will register the change in parameters for the pipeline.
4. Rerun the cross validation search with `dvc repro -s --glob cv_*`.

Please make sure to reset `LOAD_CHECKPOINTS` to `False` (and rerun `python generate_params.py`) when you want to research for hyperparameter with new data. Otherwise the search will use the old checkpoint and not rerun.

### Changing any other parameters

If you feel comfortable with diving into the code and wish to change additional parameters, you need to do the following:
1. Change the parameters in any of the python files in `src/params`.
2. Rerun `python generate_params.py` from the `scripts` folder.

If you do not complete step (2) the pipeline will not register your changes.

## Outputs from running the pipeline

These files can be found in the `results/` directory after running the pipeline.

- `predictions.csv`: Dataset used for modeling with additional columns containing predictions and probabilities for current students in Year 7-10
- `unknown_predictions.csv`: Dataset used for modeling with additional columns containing predictions and probabilities for students with unknown destinations 
- `unidentified_students_single.csv`: List of school students that had too much missing data and could not be used in the model
- `unidentified_unknowns_single.csv`: List of students with unknown destinations that had too much missing data and could not be used in the model

## Expected data schema for Power BI dashboard:

Additional datasets (`neet_annotated.csv` and `census_annotated.csv`) with data on previous years of students for the "Changes over years" Power BI dashboard page can be found in the `data/interim/` directory after running the pipeline. 

The Measures table(named as Measures_table) contains some measured valued we need to display on powerBI visualisations. We can easily create new measure in PowerBI. You will need to implement these measures (name and formula are given):
1. Att<85% 
    - Att<85% = SUM(fake_test_dataset[att_less_than_85])/DISTINCTCOUNT(fake_test_dataset[upn])

2. HighRisk
    - HighRisk = SUM(fake_test_dataset[predictions])/DISTINCTCOUNT(fake_test_dataset[upn])

3. LevelOfNeed_2%
    - LevelOfNeed_2% = SUM(fake_test_dataset[level_of_need_code_2])/DISTINCTCOUNT(fake_test_dataset[upn])

4. 	MentalHealth%
    - MentalHealth% = SUM(fake_test_dataset[characteristic_code_210])*100/DISTINCTCOUNT(fake_test_dataset[upn])

5. Pregnant/Parent%
    - Pregnant/Parent% = SUM(fake_test_dataset[Parent/Preg%])/DISTINCTCOUNT(fake_test_dataset[upn])

6. 	SEND%
    - SEND% = SUM(fake_test_dataset[send_flag])/DISTINCTCOUNT(fake_test_dataset[upn])

7. SupByYOT%
    - SupByYOT% = SUM(fake_test_dataset[characteristic_code_170])/DISTINCTCOUNT(fake_test_dataset[upn])

8. unidentified%
    - unidentified% = DISTINCTCOUNT(Unidentified[UPN])*100/DISTINCTCOUNT(fake_test_dataset[upn])


We also need to create few new columns for PowerBI. These are as follows along with the formula:
1.	Column Name: MentalHealthFlag
    - File: desens_sdv__neet_annotated
    - Formula: MentalHealthFlag = if(desens_sdv_neet_annotated[characteristic_code]="210",1,0)

2.	Column Name: Age
    - File: fake_test_dataset
    - Formula: “The Council will have to map and fill the ages”

3.	Column Name: Attendance%
    - File: fake_test_dataset
    - Formula: Attendance% = (1-fake_test_dataset[total_absences])*100

4.	Column Name: Gender
    - File: fake_test_dataset
    - Formula: Gender = IF(fake_test_dataset[gender_f]==1, "F","M")

5.	Column Name: Parent/Preg%
    - File: fake_test_dataset
    - Formula: Parent/Preg% = IF(OR(fake_test_dataset[characteristic_code_120]==1, fake_test_dataset[characteristic_code_180]==1), 1, 0)

6.	Column Name: Gender
    - File: unknowns_prediction
    - Formula: Gender = IF(unknowns_prediction[gender_m]==1, "M","F")

NOTE: replace fake_test_dataset with the actual file name which contains the predictions


@Vanshika this should not be in the final product

