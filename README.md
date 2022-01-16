# Credit_Risk_Analysis

## Background

Jill commends you for all your hard work. Piece by piece, you’ve been building up your skills in data preparation, statistical reasoning, and machine learning. You are now ready to apply machine learning to solve a real-world challenge: credit card risk.

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, you’ll need to employ different techniques to train and evaluate models with unbalanced classes. Jill asks you to use `imbalanced-learn` and `scikit-learn` libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, you’ll oversample the data using the `RandomOverSampler` and `SMOTE` algorithms, and undersample the data using the `ClusterCentroids` algorithm. Then, you’ll use a combinatorial approach of over- and undersampling using the `SMOTEENN` algorithm. Next, you’ll compare two new machine learning models that reduce bias, `BalancedRandomForestClassifier` and `EasyEnsembleClassifier`, to predict credit risk. Once you’re done, you’ll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

## Results

##### Oversampling

###### Naive Random Oversampling

![](https://github.com/labinskin/Credit_Risk_Analysis/blob/main/Resources/naive_random_oversampling.png)

The balanced accuracy for naive random oversampling was 66%.

The high-risk precision is 1%, with a sensitivity of 73%, and a F1 of 2%.

The low-risk precision is 100%, with a sensitivity of 60%, and a F1 of 75%.

###### SMOTE Oversampling

![](https://github.com/labinskin/Credit_Risk_Analysis/blob/main/Resources/smote_oversampling.png)

SMOTE Oversampling is fairly similar to Naive Random Oversampling.

The balanced accuracy is 66%.

The high-risk precision is 1%, with a sensitivity of 63%, and a F1 of 2%.

The low-risk precision is 100%, with a sensitivity of 69%, and a F1 of 82%.

##### Cluster Centroids

![](https://github.com/labinskin/Credit_Risk_Analysis/blob/main/Resources/clustercentroids_undersampling.png)

The balanced accuracy is 54%.

The high-risk precision is 1%, with a sensitivity of 69%, and a F1 of 1%.

The low-risk precision is 100%, with a sensitivity of 40%, and a F1 of 57%.

##### Combination (Over and Under) Sampling

![](https://github.com/labinskin/Credit_Risk_Analysis/blob/main/Resources/smoteenn.png)

The balanced accuracy is 69%.

The high-risk precision is 1%, with a sensitivity of 81%, and a F1 of 2%.

The low-risk precision is 100%, with a sensitivity of 57%, and a F1 of 73%.

##### Balanced Random Forest Classifier

![](https://github.com/labinskin/Credit_Risk_Analysis/blob/main/Resources/balanced_random_forest.png)

The balanced accuracy is 79%.

The high-risk precision is 3%, with a sensitivity of 70%, and a F1 of 6%.

The low-risk precision is 100%, with a sensitivity of 87%, and a F1 of 93%.

##### Easy Ensemble AdaBoost Classifier

![](https://github.com/labinskin/Credit_Risk_Analysis/blob/main/Resources/easy_ensemble.png)

Due to problems with the Easy Ensemble code, I was unable to run the fit or any of the numbers after effectively.

## Summary

The first four models oversampled, undersampled, and used a combo of over and under sample. All four of these had lower accuracy numbers, than balanced random forest, which had an accuracy of 79% and a better balance of precision and sensitivity scores, and a higher F1 score than the first four tests. With only having the projected numbers from the ensemble starter code, if these numbers held (93% accuracy, 9% precision on high-risk, 92% recall, and 16% F1), it would be the strongest model and the one I would recommend, as it provided the best balance between precision and recall, and the highest F1 score, which is what one would want.
