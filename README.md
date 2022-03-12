# Execution Instructions:

Install Dependencies: 
```
pip install scikit-learn
pip install numpy
pip install matplotlib
```
* Uses Poetry as the python virtual environment
Insallation instructions: https://python-poetry.org/docs/


To run the code leveraging poetry commands
------------
* To Run a python program  
  `poetry run my_python_app.py`
  
* To add a repository to poetry.  
  `poetry add <repo-name>`
  
# Execute main program
python linear_regression.py

# Input requirements
/data/deficit_test.dat
/data/deficit_train.dat

# Output files generated:
System print statments generate training statistics

Training_All_Degrees.png - Plot of training data with all degrees represented
Validation_w_Optimized_model.png - plot of validation data using optimized model
Lambda_For_D12.png - plot of degree 12 for different lambda values