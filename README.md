# Setting up your machine

This part will change slightly depending on what operating system you are using.

## Windows

1. Ensure you have python installed on your machine. You can install it through the [Microsoft Store](https://www.microsoft.com/store/productId/9PJPW5LDXLZ5).
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
