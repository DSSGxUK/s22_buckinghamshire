# Predicting students at risk of becoming NEET (Not in Education, Employment or Training)

Welcome to the code repository for the project conducted under **Data Science for Social Good- UK 2022 (DSSGx UK)**, for our partner: **Buckinghamshire Council**. This readme provides an overview of the project, the contributors, the folder structure of the repository, and guidance on how other councils can replicate the project using their own datasets. 

Other key resources to consult include:

  - Methodology documentation - available [here](https://github.com/DSSGxUK/s22_buckinghamshire/blob/main/docs_and_images/NEET_Prediction_Methodology_FINAL.pdf)
  - Project poster - available [here](https://github.com/DSSGxUK/s22_buckinghamshire/blob/main/docs_and_images/DSSG%20Poster%20Buckinghamshire.pdf)
  - Presentation video - available [here](https://www.youtube.com/watch?v=Oi6h_F2I_UQ)
  
## Project Overview

This project was a collaboration between [Buckinghamshire Council](https://www.buckinghamshire.gov.uk/), [the EY Foundation](https://eyfoundation.com/uk/en/home.html)  and [Data Science for Social Good UK](https://warwick.ac.uk/research/data-science/warwick-data/dssgx/). 

The goal of the project was to build a model for predicting which pupils in Buckinghamshire are at high risk of becoming NEET (Not in Education, Employment or Training) in the future, using a range of different datasets such as the School Census and National Client Caseload Information System (NCCIS) database. 

The final predictive model was developed a Gradient Boosted Trees algorithm from the LightGBM package in Python and **achieved an accuracy of 92.8% and an F2-score of 47.8%**.

### Presentation video

<p align="center">
  <a href="https://www.youtube.com/watch?v=Oi6h_F2I_UQ" target="_blank"><img src="https://github.com/DSSGxUK/s22_buckinghamshire/blob/main/docs_and_images/video2.JPG" /></a>
</p>

### Partners

[Buckinghamshire Council](https://www.buckinghamshire.gov.uk/) is a unitary local authority in South-East England responsible for providing all local government services in the region and serving a population of approximately 550k people. One of the council???s statutory duties is to support young people to participate in education, employment, or training. The council provides various services to young people and works closely with local schools to support students who are most in need of support.

[The EY Foundation](https://eyfoundation.com/uk/en/home.html) tackles the barriers to employment faced by young people from low-income backgrounds. Through their programmes, they help young people unlock their potential and succeed in the workplace.

### Challenge

Between 2018 and 2020, Buckinghamshire county had a NEET rate of above 2% and Unknown destination rate of above 5% for young people aged 17 to 18. 

Studies have shown that time spent NEET can have a detrimental effect on physical and mental health, increasing the likelihood of unemployment, low wages, or low quality of work later on in life. Buckinghamshire Council wanted to identify students??? risk of becoming NEET in years 12 or 13 (ages 17-18), by the end of  years 10 or 11 (ages 14-16) so that they could target the right pupils with early intervention programmes. It is hoped that doing so will improve the life chances of those young people who receive intervention that they otherwise may not have done.

### Data

The following datasets were provided by Buckinghamshire council and used by the team to carry out the modelling and analysis. Most of the datasets follow schemas set by central government, and where available links have been provided which describe the metadata and data fields in detail.

**NCCIS Dataset** - This dataset holds information required by Local Authorities to support young people to engage in education and training. Some of the important variables captured in the dataset are student???s characteristic codes (e.g. if they are a carer, pregnant etc), activity codes, Special Educational Needs, and their level of need. The final outcome variable of whether a student is NEET or UNKNOWN is extracted from this dataset - [NCCIS_Schema](https://www.gov.uk/government/publications/nccis-management-information-requirement).

**School Census Dataset** - Provides demographic information about students, for example: Gender, Ethnicity, Age and Language, as well as other features such as whether the student receives Free School Meals (FSM) or has Special Educational Needs (SEN) - [School Census_Schema](https://www.gov.uk/guidance/complete-the-school-census/data-items-2022-to-2023).

**KS4 Dataset** - Provides information related to student???s grades, eligibility for Free School Meals and Income deprivation index - [KS4_Schema](https://explore-education-statistics.service.gov.uk/find-statistics/key-stage-4-destination-measures/2019-20#releaseHeadlines-charts).

**Attendance Dataset** - Provides data on the attendance of students along with features like termly sessions, absences, and reasons for absences (exclusions, late entry, etc) ??? [Attendance_Schema](https://www.gov.uk/guidance/complete-the-school-census/data-items-2022-to-2023).

School Information Dataset - Provides details on school areas, postcodes, and school names.

### Methods

Two pipelines were developed for the project, to prepare the data, and to manage the modelling process. A Python package called [Data Version Control](https://dvc.org/doc/api-reference) was used to develop the modelling pipeline, which provides a simple mechanism to reproduce the project outputs using different datasets.

The diagram below provides a simple overview of the end-to-end pipeline:

![alt text](https://github.com/DSSGxUK/s22_buckinghamshire/blob/main/docs_and_images/bucks_method.jpg)

For a detailed description please refer to the [project poster](https://github.com/DSSGxUK/s22_buckinghamshire/blob/main/docs_and_images/DSSG%20Poster%20Buckinghamshire.pdf) and [methodology documentation](https://github.com/DSSGxUK/s22_buckinghamshire/blob/main/docs_and_images/NEET_Prediction_Methodology_FINAL.pdf).

### Solution & Value Add

The project team built several artefacts to support the council in this objective.

A predictive model that:
  - Predicts the risk of becoming NEET for 26,592 students currently in school years 7 to 10
  - Identifies students??? key risk factors contributing to their probability of becoming NEET
  - Identifies if students with an Unknown status are likely to become NEET

A PowerBI dashboard that:
  - Allows Buckinghamshire Council to view the insights generated by the model for each student
  - Provides insights about schools and school areas with a higher rate of NEET
  - Allows the council to view insights about a larger cohort of 61,761 unique students from the NCCIS dataset (2017-2022), in school years 6 to 13.

The final predictive model was developed a Gradient Boosted Trees algorithm from the LightGBM package in Python and achieved an accuracy of 92.8% and an F2-score of 47.8%.

At a UK level, the model has the potential to identify 22% (4,193) more students per year who become NEET as compared to the already existing Risk of NEET Indicator (RONI) tool, which is used by some local authorities in the UK. When tested on the same dataset, the RONI tool achieved an accuracy of 85.5%. 

It also flags 51% fewer students who never became NEET as compared to RONI, and therefore has the potential to save significant operational costs and resources for councils across the country.

#### Dashboard

The primary output of the project was a PowerBI dashboard which shows the council the predictions, as well as various other insights such as the key risk factors contributing to the predictions, and trends over time at the local level, school-level and pupil-level. Below is a screenshot of one view from the dashboard (please note all the data in this screenshot is synthetic):

![alt text](https://github.com/DSSGxUK/s22_buckinghamshire/blob/main/docs_and_images/PowerBI_screenshot.jpg)

#### Further work

While the model does outperform existing tools such as RONI, it is anticipated that it could be further improved by including datasets and features that are known to be predictive of poor outcomes for young people. Specifically, councils are encouraged to integrate data from sources such as Early Help & Social Care, Revenues & Benefits, and from wider public services such as the NHS and the police to further improve performance.

## Contributors

  - Abhijeet Mulgund ??? [GitHub](https://github.com/abhmul), [LinkedIn](https://www.linkedin.com/in/abhijeetmulgund/) 
  - Rachel Humphries ??? [GitHub](https://github.com/bs10reh), [LinkedIn](https://www.linkedin.com/in/rehumphries/)
  - Vanshika Namdev ??? [GitHub](https://github.com/vanshu25), [LinkedIn](https://www.linkedin.com/in/vanshikanamdev/)
  - Pranjusmrita Kalita ??? [GitHub](https://github.com/Pranjusmrita), [LinkedIn](https://www.linkedin.com/in/pranjusmrita-kalita/)

In collaboration with:
  - Project Manager: Satyam Bhagwanani ??? [GitHub](https://github.com/sat899), [LinkedIn](https://www.linkedin.com/in/satyam-bhagwanani-934a243a/)
  - Technical Mentor: Mihir Mehta ??? [GitHub](https://github.com/mihirpsu), [LinkedIn](https://www.linkedin.com/in/mihir79/)

## Folder Structure

```bash

s22_buckinghamshire
????????? .dvc
???   ????????? .gitignore                                         
???   ????????? .config                              # links to the dagshub repo                                        
????????? data		                                 
???   ????????? interim	                             # stores canonicalised, annotated and merged csv files
???   ???   ????????? attendance_canonicalized_csv
???   ???   ???   ????????? .gitignore
???   ???   ????????? ccis_canonicalized_csv
???   ???   ???   ????????? .gitignore
???   ???   ????????? census_canonicalized_csv
???   ???   ???   ????????? .gitignore
???   ???   ????????? ks4_canonicalized_csv
???   ???   ???   ????????? .gitignore
|   |   ????????? .gitignore
|   ????????? processed
|   |   ????????? .gitignore
|   ????????? raw                                  # stores the original files, won't appear until you pull synthetic data
???   ???   ????????? attendance_original_csv
???   ???   ????????? ccis_original_csv
???   ???   ????????? census_original_csv
???   ???   ????????? ks4_original_csv
???   ????????? raw.dvc
|   ????????? .gitignore
????????? example_data		                         # Example data to check schema
???   ????????????raw
???   ???   ????????????secondary_schools_original.csv
???   ???   ????????????attendance_original_csv
???   ???   ???   ????????? attendance_original_jan15.csv
???   ???   ???   ????????? attendance_original_jan22.csv
???   ???   ???   ????????? attendance_original_may15.csv
???   ???   ???   ????????? attendance_original_may22.csv
???   ???   ???   ????????? attendance_original_oct15.csv
???   ???   ???   ????????? attendance_original_oct21.csv
???   ???   ????????????ccis_original_csv
???   ???   ???   ????????? ccis_original_mar16.csv
???   ???   ???   ????????? ccis_original_mar22.csv
???   ???   ????????????census_original_csv
???   ???   ???   ????????? census_original_jan17.csv
???   ???   ???   ????????? census_original_jan22.csv
???   ???   ????????????characteristics_original_csv
???   ???   ???   ????????? characteristics_original_mar22.csv
???   ???   ????????????ks2_original_csv
???   ???   ???   ????????? ks2_original_sep20.csv
???   ???   ????????????ks4_original_csv
???   ???   ???   ????????? ks4_original_sep15.csv
???   ????????? ????????? ????????? ks4_original_sep20.csv
????????? logs
???   ????????? .gitignore
????????? metrics                             # contains metrics and results related values
???   ?????????lgbm1_single.csv                 # We've kept this one preloaded
???   ?????????lgbm2_single.csv                 # Gets created when you run the hyperparam search
|   ?????????lgbm1_basic_single.csv           # Metrics for model trained only with attendance and census data
????????? models
???   ????????? final                           # final models for prediction
???   ???   ????????? model_single.pkl            # Gets created when you retrain the model (model trained with all features)
???   ???   ????????? model_single_basic.pkl      # Gets created when you retrain the model (model trained with only attendance and census features)
???   ????????? interim
????????? notebooks                           
???   ????????? convert_synthetic.ipynb                                         
???   ????????? view_csv.ipynb
????????? plots                               # stores different plots and charts
???   ?????????attendance_percent_box_plot.png
???   ?????????common_neet_traces.png
???   ?????????consequent_antecedent.png
???   ?????????lgb1_feature_importance.png
???   ?????????neet_cons_ante.png
???   ?????????neet_infrequent_traces.png
???   ?????????neet_process_map_97.html
???   ?????????not_known_common_traces.png
???   ?????????not_known_proess_map_98.html
???   ?????????notknown_consequent_antecedent.png
???   ?????????pipline.dot
???   ?????????pipline.dot.svg
???   ?????????pipline.md
???   ?????????pipline.png
???   ?????????process_map.html
???   ?????????roni_catch_model_miss_feature_importance.png
???   ?????????sankey.png
???   ?????????sankey_debug.png
???   ?????????sankey_eet.png
???   ?????????sankey_full.png
???   ?????????sankey_neet.png
???   ?????????sankey_unknown.png                                     
???   ????????? unknown_infrequent_traces.png   
????????? results                       # where the model prediction output files will be saved
???   ????????? .gitignore
???   ????????? interim
???   ???   ????????? .gitignore   
????????? scripts                       # python code files for different purposes
???   ????????? data    
???   ???   ????????? additional_data.py
???   ???   ?????????annotate_attendance_data.py
???   ???   ?????????annotate_census_data.py
???   ???   ?????????annotate_ks4_data.py
???   ???   ?????????annotate_neet_data.py
???   ???   ?????????attendance_premerge.py
???   ???   ?????????build_ccis_ks_eda_dataset.py
???   ???   ?????????canonicalize_data.py
???   ???   ?????????census_premerge.py
???   ???   ?????????feature_selection.py
???   ???   ?????????feature_selection_for_predictions.py
???   ???   ?????????ks2_filter.py
???   ???   ?????????merge_data.py
???   ???   ?????????merge_multi_upn.py
???   ???   ?????????multi_upn_categorical.py
???   ???   ?????????multiple_to_single.py
???   ???   ?????????multiple_to_single_predict.py
???   ???   ?????????neet_premerge.py
???   ???   ?????????split_covid_years.py
???   |   ????????? xl_to_csv.py
???   ????????? misc  
???   ???   ????????? .Rhistory
???   ???   ????????? bupar-analysis.r
???   ???   ????????? compute_intersections.py
???   ???   ????????? plot_sankey.py
???   ????????? model
???   ???   ????????? merge_outputs.py
???   ???   ????????? optimization_and_cv.py
???   ???   ????????? predict.py
???   ???   ????????? retrain.py
???   ???   ????????? roni_tool.py
???   ???   ????????? split_data.py
???   ???   ????????? test.py
???   ????????? dvc.lock
???   ????????? dvc.yaml
???   ????????? generate_params.py
???   ????????? params.yaml
????????? src
???   ????????? constants
???   ???   ????????? __init__.py
???   ???   ????????? attendance_constants.py
???   ???   ????????? ccis_constants.py
???   ???   ????????? census_constants.py
???   ???   ????????? ks_constants.py
???   ???   ????????? modeling_constants.py
???   ???   ????????? school_info_constants.py
???   ???   ????????? script_argument_constants.py
???   ???   ????????? shared_constants.py
???   ????????? cv
???   ???   ????????? __init__.py
???   ???   ????????? cross_validator.py
???   ???   ????????? processing.py
???   ???   ????????? search_spaces.py
???   ???   ????????? utils.py
???   ????????? params
???   ???   ????????? __init__.py
???   ???   ????????? data_pipeline_arguments.py
???   ???   ????????? data_pipeline_params.py
???   ???   ????????? filepaths.py
???   ???   ????????? model_pipeline_arguments.py
???   ???   ????????? model_pipeline_params.py
???   ????????? aggregation_utils.py
???   ????????? attendance_utils.py
???   ????????? ccis_utils.py
???   ????????? data_utils.py
???   ????????? debug_utils.py
???   ????????? error_utils.py
???   ????????? file_utils.py
???   ????????? log_utils.py
???   ????????? merge_utils.py
???   ????????? py_utils.py
???   ????????? roni.py
????????? tests
???   ????????? scripts/data
???   ???   ????????? test_annotate_neet_data.py
???   ???   ????????? test_split_data.py
???   ????????? src
???   ???   ????????? test_cv_utils.py
???   ???   ????????? test_data_utils.py
???   ???   ????????? test_file_utils.py
???   ????????? .gitignore
????????? .dvcignore
????????? .gitignore
????????? README.md
????????? requirements.R
????????? requirements.txt
????????? setup.py

```

### Brief folder description

`data` : This folder contains two sub-folders : `interim` and `raw`. After running the pipeline an additional `processed` folder will also be present. The original dataset files are stored in their dataset sub-folder within the `raw` folder e.g. `raw/attendance_original_csv` will contain the original csv files for attendance datasets. These original files will go through the data pipeline and will generate additional files which will be canonicalized (standardised formatting), annotated and merged across years, which will be stored in `interim` sub-folder. The `processed` subfolder will contain the final datasets ready to be used for modeling.
  
`metrics` : This folder contains outputs from the hyperparameter search (`lgbm1_single.csv`, `lgbm2_single.csv`, `lgbm1_basic_single.csv`) and will also contain roni tool performance results (`roni_test_results.csv` , `roni_test_results_basic.csv`) and model performance results (`single_test_results.csv`, `single_test_results_basic.csv`) on the test dataset.

`models` : This folder contains pickle files of the models. There are two sub-folders: `interim` and `final`. `interim` holds the checkpoints. You can find more details about these in the [Reloading the hyperparameter search from a checkpoint](#reloading-the-hyperparameter-search-from-a-checkpoint) section. The final, retrained best models will be found `models/final/model_single.pkl` and `models/final/model_single_basic.pkl`. `model_single.pkl` is the model trained with features from all datasets and `model_single_basic.pkl` is the model trained with only attendance and census dataset features. 

`results` : After running the pipeline, this folder will contain the final output CSV files: `predictions.csv`, `unknown_predictions.csv`, `unidentified_students_single.csv`, `unidentified_unknowns_single.csv`. These files are outlined in more detail below under [Outputs from running the pipeline](#outputs-from-running-the-pipeline)

`scripts` : This folder contains the `dvc.yaml` file that outlines the different stages and steps of the pipeline. It also includes two main sub-folders: `data` and `model`. The `data` sub-folder contains python scripts that prepare the interim datasets and final datasets used for modeling. The `model` sub-folder contains scripts that split the final dataset into train and test datasets, runs the cross-validation and hyperparameter search, re-trains the model, calculates roni scores and uses the trained model to generate predictions for current/unknown students.        

`src` : This folder contains helper functions (found in the `*_utils.py` scripts) and also contains scripts that can set different parameters. There are three sub-folders: `constants`, `cv` and `params`. The `cv` sub-folder contains helper functions for the cross-validation and hyper-parameter search stage. It also contains dictionaries of the hyper-parameter search spaces in `search_spaces.py`. The `constants` folder contains parameters for the pipeline that are unlikely to need to change, whereas the `params` sub-folder contains parameters for the pipeline that may need/want to be changed. The `*_arguments.py` scripts in this sub-folder include the arguments that are sent to the `dvc.yaml` pipeline.    

## How to Replicate the Project

### Data Assumptions

Assuming that the data provided by the user are of the following types:
  1. Attendance [Attendance_Schema](https://www.gov.uk/guidance/complete-the-school-census/data-items-2022-to-2023)
  2. Census [Census_Schema](https://www.gov.uk/guidance/complete-the-school-census/data-items-2022-to-2023)
  3. CCIS [CCIS_Schema](https://www.gov.uk/government/publications/nccis-management-information-requirement)
  4. KS4 [KS4_Schema](https://explore-education-statistics.service.gov.uk/find-statistics/key-stage-4-destination-measures/2019-20#releaseHeadlines-charts)

In addition, we allow data on *characteristics* and *ks2*, since these can be used as features for the model before the student enters year 11.

**Points to remember**:
  1. Please ensure files are in CSV format only
  2. Currently columns are renamed to `snake_case` (lowercase with spaces as _). We suggest first you try to make your column names match the schemas in the `example_data/raw` folder. If you instead want to change the columns the code can process, you'll need to add entries to the renaming dictionary. You can find the renaming dictionary in the `src/constants/[TYPE]_constants.py` file where `[TYPE]` refers to whatever your data type is. The *ks2* renaming dictionary is in `src/constants/ks_constants.py` and the *characteristics* renaming dictionary is the same as the CCIS renaming dictionary in `src/constants/ccis_constants.py`. After renaming columns please run `python ./generate_params.py` from your `scripts` folder. See [Changing Any Other Parameters](#changing-any-other-parameters) for more details on how to do this.
  3. We assume that the CCIS datasets have a `month/year of birth` column with numeric month and year values in the form `[MONTH]/[YEAR]`. We don't use the date of
     birth column. You can safely remove it if there is concern about data sensitivity.
  4. No column names can include "__"! This is a special reserved character for our code.


### Setting up a machine for running all the workflows

This part will change slightly depending on what operating system you are using.

#### Windows

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
# Set-ExecutionPolicy RemoteSigned -Scope CurrentUser                 # We can do this by running the command to the
                                                                      # left without the #. You may need
                                                                      # administrator privelages to do this.
                                                              
> python.exe -m pip install --upgrade pip                             # Update pip if necessary

> python.exe -m pip install -r .\requirements.txt                     # Install required python packages

> dvc config --system core.analytics false                            # Turn off DVC anonymized analytics

```

#### Mac or Linux

1. Ensure you have an updated python installed on your machine. For Mac, you can find the python installer at [python.org](https://www.python.org/downloads/macos/). For Linux, you can find the installer [here](https://www.python.org/downloads/source/)
2. Ensure you have git installed. For Mac, you can get an installer [here](https://git-scm.com/download/mac). For Linux, you can find directions for installing [here](https://git-scm.com/download/linux)
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

> source ./venv/bin/activate                                      # Running the virtual environment. 
                                                              
> pip install --upgrade pip                                           # Update pip if necessary

> pip install -r .\requirements.txt                                   # Install required python packages

> dvc config --system core.analytics false                            # Turn off DVC anonymized analytics
```

### Connecting Data

#### Downloading the synthetic data

We've published synthetic data to dagshub so you can play around with the pipeline. This data is randomly generated and any unique IDs are random strings.

 To retrieve it, please run the following
```bash
dvc remote add origin https://dagshub.com/abhmul/s22_buckinghamshire.dvc -f
dvc pull -r origin
```

#### Using your own data

If you are a council with your own data, these datasets will need to be saved in the `data/raw` directory as csv files in the correct formats with the correct column names. 

Before adding your data, please run the steps above to get the synthetic data. Then run the following steps

##### Windows 

```bash
cd data/raw
dvc remove remove origin                                  # Run this without # if you downloaded the synthetic data
Get-ChildItem * -Include *.csv -Recurse | Remove-Item     # Run this without # to remove any synthetic data have
                                                          # Please note this deletes all csv files in `data/raw`
Get-ChildItem * -Include *.csv.dvc -Recurse | Remove-Item # This deletes the dvc tracking files for the synthetic data
```

##### Linux or Mac

```bash
cd data/raw
dvc remove remove origin          # Run this without # if you downloaded the synthetic data
rm **/*.csv && rm *.csv           # Run this without # to remove any synthetic data have
                                  # Please note this deletes all csv files in `data/raw`
rm **/*.csv.dvc && rm *.csv.dvc   # This deletes the dvc tracking files for the synthetic data
```

For an example of what the schema of the datasets and folder structure should look like, we've kept snippets of synthetic data for you to compare against in the `example_data/raw` folder.

Within the `data/raw` directory are 6 folders that correspond to the different datasets listed above under [Assumptions](#assumptions): 
- `attendance_original_csv`
- `ccis_original_csv`
- `census_original_csv`
- `ks4_original_csv`
- `characteristics_original_csv`
- `ks2_original_csv`

The datasets in these directories should be named `[TYPE]_original_[DATE].csv` where `[TYPE]` refers to the dataset (attendance, ccis, census, ks4, characteristics, ks2) and `[DATE]` refers to the month and year the dataset was submitted (e.g. `attendance_original_jan21.csv` corresponds to autumn 2021 attendance data, which is submitted in January). `[DATE]` should be written as the first 3 letters of the month and the last 2 digits of the year e.g. `jan21`, `sep19`.

CSV files in `characteristics_original_csv` and `ks2_original_csv` contain columns from the CCIS and KS4 datasets, respectively, and should be populated with data from current Year 7-10 students that we want to generate predictions for. These are separate datasets as current Year 7-10 students are not present in the CCIS and KS4 datasets until Year 11 onwards. Currently, predictions are generated for students without this additional data using the model trained only with attendance and census data (`model_single_basic.pkl`). However, the model trained with features from all the datasets (`model_single.pkl`) performs better on the test dataset and therefore including these additional characterisitcs and ks2 datasets for current students will improve the accuracy of the predictions.

In addition you should add a csv file called `data/raw/secondary_schools_original.csv`, so the code knows what schools the establishment numbers in the data correspond to. See the file in `example_data/raw` for how your csv should look.

Once you've added your data to the `data/raw` folder, you should be good to go.

#### Adding data from new years

You may want to incorporate more data from later years as you collect it. Simply follow the procedure outlined above, and the pipeline will pick it up.

#### Adding New Columns

We currently do not support addition of new columns for modeling. The code should work fine if you add new columns but it will not use them in modeling.

### Running the code

**Please follow the below steps before running the workflows**:
  
```bash
cd .\scripts\
```

#### Run the whole pipeline

Running the whole pipeline includes a hyper parameter search which can take a few hours to complete. If you do not wish to run this stage, please follow the instructions under [Output predictions on new data without re-running the hyper parameter search](#output-predictions-on-new-data-without-re-running-the-hyper-parameter-search).

To run the whole pipeline you can run:

```bash
dvc repro
```

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

  #### Output predictions on new data without re-running the hyper parameter search  
  
  Following these steps re-trains the model with new data using the previous best hyper-parameters.
    
```bash
  # Generate datasets for modelling
  dvc repro -s --glob generate_modeling_*          

  # Model Evaluation 
  dvc repro -s --glob model_evaluation_*                        

  # Generate datasets for predictions and final output
  dvc repro -s --glob prediction_* 
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

#### Reloading the hyperparameter search from a checkpoint

Because the hyperparameter search takes a long time, we have built support for checkpoint progress. If for some reason the run does not complete, you can pick it up from where it left off rather than restarting it. To do this you will need to complete the following steps:
1. Open `src/params/model_pipeline_params.py`.
2. Find the variable `LOAD_CHECKPOINTS`. Change its value to `True`.
3. From the `scripts` folder (you may already be there if you were running the pipeline), run `python generate_params.py`. This will register the change in parameters for the pipeline.
4. Rerun the cross validation search with `dvc repro -s --glob cv_*`.

Please make sure to reset `LOAD_CHECKPOINTS` to `False` (and rerun `python generate_params.py`) when you want to research for hyperparameter with new data. Otherwise the search will use the old checkpoint and not rerun.

#### Changing any other parameters

If you feel comfortable with diving into the code and wish to change additional parameters, you need to do the following:
1. Change the parameters in any of the python files in `src/params`.
2. Rerun `python generate_params.py` from the `scripts` folder.

If you do not complete step (2) the pipeline will not register your changes.

#### Outputs from running the pipeline

These files can be found in the `results/` directory after running the pipeline.

- `predictions.csv`: Dataset used for modeling with additional columns containing predictions and probabilities for current students in Year 7-10
- `unknown_predictions.csv`: Dataset used for modeling with additional columns containing predictions and probabilities for students with unknown destinations 
- `unidentified_students_single.csv`: List of school students that had too much missing data and could not be used in the model
- `unidentified_unknowns_single.csv`: List of students with unknown destinations that had too much missing data and could not be used in the model

#### Expected data schema for Power BI dashboard:

The datasets to be loaded into the Power BI dahsboard after running the pipeline are found here:
- `results/predictions.csv` 
- `results/unknown_predictions.csv`
- `results/unidentified_students_single.csv`
- `data/interim/neet_annotated.csv`
- `data/interim/census_annotated.csv`
- `data/interim/attendance_exact.csv`

The first two contain predictions for current school students in Years 7-10 and for current unknown students, respectively.
`unidentified_students_single.csv` contains unidentified current school students for which predictions could not be generated due to too much missing data.
The final three files found in the `data/interim` folder contain neet, census and attendance data from previous years of students. These three datasets are for a separate page of the power bi dashboard that looks at certain factor trends over the years.    

The Measures table(named as Measures_table) contains some measured valued we need to display on powerBI visualisations. We can easily create new measure in PowerBI. You will need to implement these measures (name and formula are given) in case it is gone:
- Att<85% 
    `Att<85% = SUM(predictions[att_less_than_85])/DISTINCTCOUNT(predictions[upn])`

- HighRisk
    `HighRisk = SUM(predictions[predictions])/DISTINCTCOUNT(predictions[upn])`

- LevelOfNeed_2%
    `LevelOfNeed_2% = SUM(predictions[level_of_need_code__2])/DISTINCTCOUNT(predictions[upn])`

- MentalHealth%
    `MentalHealth% = SUM(predictions[characteristic_code__210])*100/DISTINCTCOUNT(predictions[upn])`

- Pregnant/Parent%
    `Pregnant/Parent% = SUM(predictions[Parent/Preg%])/DISTINCTCOUNT(predictions[upn])`

- SEND%
    `SEND% = SUM(predictions[send_flag])/DISTINCTCOUNT(predictions[upn])`

- SupByYOT%
    `SupByYOT% = SUM(predictions[characteristic_code__170])/DISTINCTCOUNT(predictions[upn])`
    
- unidentified%
    `unidentified% = DISTINCTCOUNT(unidentified_students_single[UPN])/DISTINCTCOUNT(predictions[upn])`
    
- Exclusions%
    `Exclusions% = AVERAGE(predictions[excluded_authorised_percent1])*100`


We also need to create few new columns for PowerBI. These are as follows along with the formula:
1.	Column Name: `mental_health_flag`
    - File: `neet_annotated`
    - Formula: `mental_health_flag = IF(neet_annotated[characteristic_code]==210, 1,0)`

2.	Column Name: `Attendance%`
    - File: `predictions`
    - Formula: `Attendance% = (1-predictions[total_absences])*100`

3.	Column Name: `Gender`
    - File: `predictions`
    - Formula: `Gender = SWITCH(TRUE(), 'predictions'[gender__f]==1, "F", 'predictions'[gender__m]==1, "M", 'predictions'[gender__u]==1, "U", 'predictions'[gender__W]==1, "W")`

4.	Column Name: `Parent/Preg%`
    - File: `predictions`
    - Formula: `Parent/Preg% = IF(OR(predictions[characteristic_code__120]==1, predictions[characteristic_code__180]==1), 1, 0)`

5.	Column Name: `Gender`
    - File: `unknown_predictions`
    - Formula: `Gender = SWITCH(TRUE(), 'unknown_predictions'[gender__f]==1, "F", 'unknown_predictions'[gender__m]==1, "M", 'unknown_predictions'[gender__u]==1, "U", 'unknown_predictions'[gender__W]==1, "W")`
    
    
 `Relationships`
 - For the map visualisations, you'll need to create two relationships. For this you'll have to upload the file 'PCD_OA_LSOA_MSOA_LAD_NOV19_UK_LU.csv'.
 - First relationship, you will need to create relationship between column `pcds` of file `PCD_OA_LSOA_MSOA_LAD_NOV19_UK_LU.csv` with the column `postcode` of file `predictions`.
 - Another relationship you will need to create is between column `pcds` of file `PCD_OA_LSOA_MSOA_LAD_NOV19_UK_LU.csv` with the column `postcode` of file `unknown_predictions`.
