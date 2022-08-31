# Predicting students at risk of becoming NEET (Not in Education, Employment or Training)

Welcome to the code repository for the project conducted under **Data Science for Social Good- UK 2022 (DSSGx UK)**, for our partner: **Buckinghamshire Council**. The aim of the project was to identify students, before Year 11, at risk of becoming NEET after they complete their GCSEs. This readme will focus on documenting:

1.	Folder structure 
2.  Assumptions
3.	Setting up a machine for running all the workflows
4.	How to run different workflows
5.	Expected data schema for Power BI dashboard

# Folder Structure

@Vanshika - this should be written out in full sentences and explained
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
3. Open a powershell as administrator and run the below steps:
>
```bash
> cd                                                                  # Navigate to the desired parent directory using this (change directory) command. 

> ls                                                                  # You can run this command to see the contents of a directory 

> git clone https://github.com/DSSGxUK/s22_buckinghamshire.git        # This will download the code repository to the current folder

> cd s22_buckinghamshire                                              # This navigates to the repository folder

> python -m venv venv                                                 # Creating a python virtual environment

> .\venv\Scripts\activate                                             # Running the virtual environment. 
                                                                      # If you get an error that says '... s22_buckinghamshire\venv\Scripts\Activate.ps1'
                                                                      # cannot be loaded because running scripts is disabled on this system. For more information,
                                                                      # see about_Execution_Policies at 'https:/go.microsoft.com/fwlink/?LinkID=135170',
                                                                      # then we need to enable execution of signed scripts. 
                                                                      # We can do this by running 'Set-ExecutionPolicy RemoteSigned -Scope CurrentUser'.
                                                              
> python.exe -m pip install --upgrade pip                             # Update pip if necessary

> pip install -r .\requirements.txt                                   # Install required python packages

```

# How to run different workflows

## Downloading the synthetic data

We've published synthetic data (data that does not come from any real person) to dagshub so 
you can play around with the pipeline. To retrieve it, please run the following
```bash
dvc remote add origin https://dagshub.com/abhmul/s22_buckinghamshire.dvc
dvc pull -r origin
```

## Using your own data

If you are a council with your own data, these datasets will need to be saved in the `data/raw` directory as csv files in the correct formats with the correct column names. We have examples of what this should look like here...

Within the `data/raw` directory are 4 folders that correspond to the different datasets listed above under *Assumptions*: 
- `attendance_original_csv`
- `ccis_original_csv`
- `census_original_csv`
- `ks4_original_csv`

The datasets in these directories should be named `[TYPE]_original_[DATE].csv` where `[TYPE]` refers to the dataset (attendance, ccis, census, ks4) and `[DATE]` refers to the month and year of the dataset (e.g. `attendance_original_jan21.csv`). `[DATE]` should be written as the first 3 letters of the month and the last 2 digits of the year e.g. `jan21`, `sep19` 

**Please follow the below steps before running the workflows**:
  
```bash
cd .\scripts\
```

## Run the whole pipeline
To run the whole pipeline you can just run:

```bash
dvc repro
```

Alternatively, you could run the individual steps:

```bash
  # Generate datasets for modelling
  dvc repro -s --glob generate_modeling_*

  # Run cross validation and hyper parameter search 
  dvc repro -s --glob cv_*

  # Model Evaluation 
  dvc repro -s --glob evaluate_model_* 

  # Generate datasets for predictions and final output 
  dvc repro -s --glob prediction_* 
```
    
  ## Run the prediction using a model trained on older data
    
  ```bash
    # Generate datasets for modelling
    dvc repro -s --glob generate_modeling_*          
    
    # Model Evaluation 
    dvc repro -s --glob evaluate_model_*                        
    
    # Generate datasets for predictions and final output
    dvc repro --glob prediction_* 
   
  ```
    
  ## Run the old model with new data
    
  ```bash
    # Generate datasets for modelling
    dvc repro -s --glob generate_modeling_*          
    
    # Retrain model 
    dvc repro --glob retrain_*
    
    # Model Evaluation 
    dvc repro -s --glob evaluate_model_*                        
    
    # Generate datasets for predictions and final output
    dvc repro --glob prediction_* 
   
  ```

Below is a brief overview of what each stage within a workflow is doing:

**Generate datasets for modelling**
  - Merges each dataset (eg: Census, Attendance,etc) across all the years 
  - Split categorical variables into binary columns containing 0's or 1's 
  - Drop columns which aren't required for modelling or for which we won't have data available before Year 11
  - Output two datasets ready for modelling: 
    - Only unique students (UPNs)
    - A student having multiple observations
    
**Run cross validation and hyper parameter search**
  - Searches for the best model parameters. Please note you can opt out of running this step if you want. 
  
**Retrain Model**
  - Re-trains the model with the new incoming data and outputs model with best performance
  
**Model Evaluation**
  - Evaluates RONI tool's performance
  - Retrains and saves model with best threshold
  - Apply the chosen model on the test data
  
**Generate datasets for predictions and final output**
  - Creates datasets required for final predictions
  - Executes model on unseen data and generates final predictions in form of a CSV
  - Generates feature importance
  - Returns RONI score
  - Returns scaled probability scores for a student at risk of becoming NEET (between 1-10)

@Vanshika - we need directions on how to rerun the hyperparameter search from a checkpoint. To do this, the user has to change the LOAD_CHECKPOINTS value in the params.yaml file to true. When they rerun the pipeline with new data, they should set this to false otherwise it will use an old checkpoint for new data.

## Outputs from running the model

- `predictions.csv`: Contains the dataset used for modeling with additional columns containing predictions and probabilities for current students in Year 7-10
- `unknown_predictions.csv`: Contains the dataset used for modeling with additional columns containing predictions and probabilities for students with unknown destinations 
- `unidentified_students_single.csv`: List of school students that had too much missing data and therefore could not be used in the model
- `unidentified_unknowns_single.csv`: List of students with unknown destinations that had too much missing data and therefore could not be used in the model
- annotated_ccis_data
- annotated_census_data

## Expected data schema for Power BI dashboard:

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

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Some initial notes as we move towards a README:
1. We are now assuming that the user provides all the data they have of the following 4 types:
  1. Attendance
  2. Census
  3. CCIS
  4. KS4
  
  In addition, we want to allow data on *characteristics* and *ks2*. This has not been supported yet,
  but would fill in features we are passing to the model for training.
2. If files come in as excel (.xlsx), they must be one sheet per file. We do not support multiple sheets per
excel file. It may just be easier to export your files directly as CSV.
3. Currently columns are renamed to `snake_case`. You may need to add more columns to the renaming dictionary if
your columns have changed or are different. You can find the renaming dictionary in the `src` directory in
the `[TYPE]_utils.py` file where `[TYPE]` refers to whatever your data type is. Note ks2 and characteristics
will not currently show up in there.
4. We assume that the CCIS datasets have a `month/year of birth` column with numeric month and year values in the form
`[MONTH]/[YEAR]`. We don't use the date of birth column. You can safely remove it if there is concern about data sensitivity.

NOTE: No column names can include "__"! This is a special reserved character for our code.

TODO for presentation
- Setup a python venv
- Install dvc - install from pip requirements
- Workflows:
  - Run the whole pipeline 
  - Run the train w/ old best params
  - Run the old model with new data
- Include links from government dataset.
- list of names of all files that they need.
  - prediction dataset
  - unknown predictions
  - unidentifiied students
  - annotated_ccis_data
  - annotated_census_data
    - merge with annotated_ccis_data?
    

