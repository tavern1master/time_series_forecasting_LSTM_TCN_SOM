Folder models contains results of LSTM and TCN training and parametrization. There are parameters and weight matrices for forecasting models.
Folder data should contain only raw data, external data (SOM_8Neurons_15Epochs_Summary_pickle) and preprocessed data after running  1_Manuscript notebook or run.py
Folder notebook consists of one notebook with code example.
Folder reports contains results of experiments after running 1_Manuscript or run.py.

Setup all dependencies in virtual environment
Windows:
`python -m venv venv`
`venv\Scripts\activate`
Unix: 
`python3 -m venv venv`
`source venv/bin/activate`

1. Install all necessary dependencies from requirements.txt `pip install -r requirements.txt`
2. Add dataset to data/raw AND add to data/external from https://github.com/sebastianachter/Clustering_for_data_generation -> results/SOM/SOM_8Neurons_15Epochs_Summary_pickle
3. Change filename in src/config.py FNAME
4. Run in terminal run.py. ATTENTION: in case of any errors, pleaso go to step 5. 
5. Open jupyter notebook and inspect 1_Manuscript for code details
6. Results of experiments can be obtained in reports/tables. Data transformation is in data/processed

Training of models is not included in run.py. In order to access model training details, please go to src/features/training_pipeline.py