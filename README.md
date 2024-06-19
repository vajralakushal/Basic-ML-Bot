# Basic-ML-Bot

This is a script that I utilize to help speed up a basic machine learning workflow. Keep in mind that this script is intended for primarily tabular data where either regression or classification is intended for a target column. The script determines whether classification or regression should be done on the dataset, then it iterates through a list of models, training each of them and further tunes the hyperparameters to find the best ones. Once the best model for the use case has been found, the script outputs a Python file which contains the code necessary to achieve similar performance with the highest performance models.

The intended usage is like this: 
```
python basic-ml-bot.py <path_to_csv> <target_column>
```

This README needs more work, but here's the to-do:

### To-Do
- Deploy on personal website
- Implement LightGBM/CatBoost whenever I feel like it