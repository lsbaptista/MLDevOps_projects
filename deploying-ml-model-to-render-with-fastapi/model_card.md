# Model Card

For additional information see the Model Card paper: [Model Card Paper](https://arxiv.org/pdf/1810.03993.pdf)

## Model Details

This model is a Logistic Regression classifier, created as part of a machine learning pipeline defined in the starter/starter project structure. The model uses the default hyperparameters from scikit-learn (>= 1.6.1). It was trained to predict whether an individual's income exceeds $50K based on demographic and employment features from the U.S. Census dataset. The project includes code for preprocessing the data, training, evaluating, and persisting the model.

## Intended Use

This model is intended to predict whether an individual's income exceeds $50K or not, based on features such as age, education, occupation, and other demographic and employment attributes. This model is useful for researchers, analysts, and policymakers who need to quickly classify individuals into income categories.

**Intended users**: Data scientists, analysts, or automated systems requiring binary income-level classification for individuals based on census data.

## Training Data

- **Source**: UCI Adult Census Income Dataset
- **Features**: age, workclass, fnlgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country
- **Label**: salary (<=50K or >50K)
- **Preprocessing**: 
  - Categorical features were encoded using OneHotEncoder.
  - The label (salary) was encoded using LabelBinarizer.
  - Continuous features were retained as-is.
- **Train/Test Split**: K-Fold Cross Validation: The dataset was divided into 5 folds for cross-validation using StratifiedKFold from scikit-learn. This method ensures that each fold maintains the same proportion of classes (i.e., the "salary" label) as the original dataset, enhancing the model's ability to generalize. Each fold is used as a validation set once, with the remaining data used for training, allowing for more robust evaluation.

## Evaluation Data

The model's performance was evaluated using K-Fold Cross Validation with 5 splits. This method ensures that the model is tested on different subsets of the data multiple times, which helps in assessing its ability to generalize across various unseen data distributions. The stratification ensures that each fold contains a balanced distribution of demographic and employment features, providing a thorough evaluation of the model.

# Cross-Validation Summary

The model's performance across the K-Fold splits is summarized by the following metrics:

- **Average Precision**: 0.7336
- **Average Recall**: 0.6259
- **Average F1 Score**: 0.6755

These metrics indicate that the model is fairly good at distinguishing individuals with higher income from those with lower income, but there is still room for improvement, particularly in terms of recall.

## Metrics

The model was evaluated using the following metrics, with the **mean value calculated across all K-Fold splits**:

- **Precision**: Measures the proportion of positive predictions that were correct, averaged across all folds.
- **Recall**: Measures the proportion of actual positives that were correctly identified by the model, averaged across all folds.
- **F1 Score**: The harmonic mean of precision and recall, offering a balance between the two, calculated as the mean across all folds.

These metrics help ensure that the model maintains a balance between identifying individuals with high income (positive class) and avoiding false positives or negatives. By computing the mean across the K-Fold splits, the evaluation becomes more robust and reliable across different subsets of the data.

## Ethical Considerations

This model may inherit biases from the data it was trained on, particularly with respect to:
- **Socioeconomic and racial biases**: The dataset contains inherent societal biases that may lead to skewed predictions for certain demographic groups.
- **Gender bias**: The data may reflect gender imbalances in certain occupations and income levels, potentially influencing the model's predictions.

It is recommended to use fairness audit tools such as Aequitas or Fairlearn to analyze and mitigate any disparities that might arise from these biases.

## Caveats and Recommendations

- **Data Quality**: The model's performance heavily depends on the quality and representativeness of the training data. If the data is outdated or unbalanced, the model's predictions may be less accurate.
- **Feature Engineering**: While continuous features were kept as-is, additional preprocessing or scaling could improve the model's performance, particularly if other machine learning models are used.
- **Bias**: As mentioned earlier, care should be taken to evaluate and address any potential biases, especially if the model is deployed in a real-world system where fairness is a concern.
- **Use Case**: This model is designed for binary income classification. It may not generalize well to other tasks, especially in different domains or datasets.


