# data_outputs directory

This directory contains the output data from the data preperation code, acting as inputs to the model training code.

File names are as follows:

- First part (train, val, test) indicates the data split the file belongs to.

- Second part (fold_i for i from 1 to 3) for train and val data indicates the fold the data in the file belongs to. In this project, due to the computational cost of training, we only choose fold 1, but we leave the data for all folds for future work to also evaluate the model on the other folds.

- Third part (n_j for j being either 5, 10, or 20) indicates the $N$ value (length of subsequence) of the specific data in the file.

All JSON files include a list of dictionaries, where each dictionary represents a subsequence. The data is separated into train, validation, and test sets based on student ID.