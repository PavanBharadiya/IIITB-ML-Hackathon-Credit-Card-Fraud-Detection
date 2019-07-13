Credit Card Fraud Detection by
1.Bharadiya Pavan Vijaykumar(MT2018023)
2.Devarakonda Srinivas Deepak(MT2018031)


Packages required:
imbalanced-learn (0.4.3)
imblearn (0.0)
matplotlib (3.0.2)
numpy (1.15.4)
pickleshare(0.7.5)
pandas (0.23.4)
pandas-ml (0.5.0)
seaborn (0.9.0)
sklearn (0.0)

Description:

We have used 4 models(LogisticRegression,DecisionTree,RandomForest,GradientBoosting) for classification.

1.Logistic Regression gave the best results.We have three different models for LR.
	a) First is without oversampling. The model is pickled in 'basic_LR.sav'.
	b) Second is with oversampling the class 1. The model is pickled in 'over_sampled_LR.sav'.
	c) Third is with oversampling with balanced class weights. The model is pickled in 'over_sampled_LR_with_balanced_class_weight.sav'.

2. DecisionTree model is pickled in 'over_sampled_DT.sav'.
3. RandomForest model is pickled in 'over_sampled_RF.sav'.
4. GradientBossting is pickled in 'over_sampled_GB.sav'.


To see the results:

1.Unpickle 'over_sampled_LR_with_balanced_class_weight.sav'.
2.Load and split the 'creditcard.csv' into features and label.
3.Give them to unpickled model and check the accuracy.
4.Build Confusion Matrix for the model and check FNR.
You will get the best results using the above unpickled model having accuracy = 0.9757 and FNR = 0.1011,ROC = 0.94
You can check for other models as well where accuracy is somewhat similar but FNR is around 0.18.
