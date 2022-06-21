# Example: generate features for a sample csv file. Make sure ../new_features dir is created
python3 features_generation.py --csvfile ./sample_data_stand.csv --model CPE --split-type te --ft ecfp0 --numpy_folder ../new_features
