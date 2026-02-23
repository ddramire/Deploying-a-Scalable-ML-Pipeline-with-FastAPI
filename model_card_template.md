# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model predicts whether a person earns more than $50K per year using U.S. Census data.
The model type is Logistic Regression from scikit-learn.
Categorical features were encoded using OneHotEncoder, and the salary label was converted to binary using LabelBinarizer.
The trained model and encoder are saved as model.pkl and encoder.pkl.

## Intended Use
This model was created for educational purposes as part of a machine learning project.
It demonstrates how to build a full ML pipeline, including data processing, model training, evaluation, and slice-based performance analysis.
It is not intended to be used for real-world decisions such as hiring, lending, or government benefits.

## Training Data
The model was trained using the Adult Income dataset from the U.S. Census.
The dataset includes demographic and employment-related features such as age, workclass, education, marital status, occupation, race, sex, and native country.
The data was split into 80% training data and 20% testing data.
Categorical variables were one-hot encoded before training.

## Evaluation Data
The model was evaluated on the 20% test split of the same dataset.
The same preprocessing steps used for the training data were applied to the test data using the fitted encoder and label binarizer.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

The model was evaluated using Precision, Recall, and F1 Score.
These metrics were chosen because they measure how well the model correctly identifies income classes.

Model performance on the test set:
    Precision: 0.7343
    Recall: 0.5746
    F1 Score: 0.6447

Precision measures how many predicted high-income cases were correct.
Recall measures how many actual high-income cases were correctly identified.
F1 Score balances precision and recall.

Model performance was also evaluated across categorical slices. The results are saved in slice_output.txt.

## Ethical Considerations
The dataset includes sensitive attributes such as race and sex.
The model uses demographic information to predict income, so it may reflect existing societal biases in the data.
This model should not be used for real-world decision-making that affects individuals.

## Caveats and Recommendations
The model was trained using Logistic Regression without extensive hyperparameter tuning.
A convergence warning occurred during training, but the model still completed successfully.
Performance may improve with additional tuning or feature scaling.
The model was evaluated only on data from the same dataset, so performance on other populations is unknown.
This model is intended for educational use only.
