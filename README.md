# Setting up your machine

This part will change slightly depending on what operating system you are using.

## Windows

1. Ensure you have an updated python installed on your machine. You can install it through the [Microsoft Store](https://www.microsoft.com/store/productId/9PJPW5LDXLZ5). As of writing this, the most up to date version was python 3.10.
2. Ensure you have git installed. You can get an installer [here](https://git-scm.com/download/win).
3. Open a powershell as administrator
4. Navigate to the desired parent directory using the `cd` (change directory) command. You can run `ls` to see the contents of a directory.
5. Run `git clone https://github.com/DSSGxUK/s22_buckinghamshire.git`. This will download the code repository to the current folder.
6. Run `cd s22_buckinghamshire` to navigate to the repository folder.
7. Create a python virtual environment by calling `python -m venv venv`. 
8. Run the virtual environment by calling `.\venv\Scripts\activate`. If you get an error that says ```... s22_buckinghamshire\venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system. For more information, see about_Execution_Policies at https:/go.microsoft.com/fwlink/?LinkID=135170```, then we need to enable execution of signed scripts. We can do this by running `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`.
9. Update pip if necessary `python.exe -m pip install --upgrade pip`.
10. Install required python packages `pip install -r .\requirements.txt`.

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
    
    
## PowerBi Notes:
•	The Measures table(named as Measures_table) contains some measured valued we need to display on powerBI visualisations. We can easily create new measure in PowerBI. You will need to implement these measures (name and formula are given):
a.	Att<85% 
Att<85% = SUM(fake_test_dataset[att_less_than_85])/DISTINCTCOUNT(fake_test_dataset[upn])

b.	HighRisk
HighRisk = SUM(fake_test_dataset[predictions])/DISTINCTCOUNT(fake_test_dataset[upn])

c.	LevelOfNeed_2%
LevelOfNeed_2% = SUM(fake_test_dataset[level_of_need_code_2])/DISTINCTCOUNT(fake_test_dataset[upn])

d.	MentalHealth%
MentalHealth% = SUM(fake_test_dataset[characteristic_code_210])*100/DISTINCTCOUNT(fake_test_dataset[upn])

e.	Pregnant/Parent%
Pregnant/Parent% = SUM(fake_test_dataset[Parent/Preg%])/DISTINCTCOUNT(fake_test_dataset[upn])

f.	SEND%
SEND% = SUM(fake_test_dataset[send_flag])/DISTINCTCOUNT(fake_test_dataset[upn])

g.	SupByYOT%
SupByYOT% = SUM(fake_test_dataset[characteristic_code_170])/DISTINCTCOUNT(fake_test_dataset[upn])

h.	unidentified%
unidentified% = DISTINCTCOUNT(Unidentified[UPN])*100/DISTINCTCOUNT(fake_test_dataset[upn])



•	We also need to create few new columns for PowerBI. These are as follows along with the formula:
a.	Column Name: MentalHealthFlag
File: desens_sdv__neet_annotated
Formula: MentalHealthFlag = if(desens_sdv_neet_annotated[characteristic_code]="210",1,0)

b.	Column Name: Age
File: fake_test_dataset
Formula: “The Council will have to map and fill the ages”

c.	Column Name: Attendance%
File: fake_test_dataset
Formula: Attendance% = (1-fake_test_dataset[total_absences])*100

d.	Column Name: Gender
File: fake_test_dataset
Formula: Gender = IF(fake_test_dataset[gender_f]==1, "F","M")

e.	Column Name: Parent/Preg%
File: fake_test_dataset
Formula: Parent/Preg% = IF(OR(fake_test_dataset[characteristic_code_120]==1, fake_test_dataset[characteristic_code_180]==1), 1, 0)

f.	Column Name: Gender
File: unknowns_prediction
Formula: Gender = IF(unknowns_prediction[gender_m]==1, "M","F")




NOTE: replace fake_test_dataset with the actual file name which contains the predictions

