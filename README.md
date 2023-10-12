# Seed Dataset Classification and Clustering

## Table of Contents
- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
  - [Data Import](#data-import)
  - [Data Representation](#data-representation)
  - [Correlation Analysis](#correlation-analysis)
  - [Data Normalization](#data-normalization)
- [Correlation Heatmap](#correlation-heatmap)
- [Data Pre-processing](#data-pre-processing)
- [Train and Test Sets](#train-and-test-sets)
- [Classification Algorithms](#classification-algorithms)
  - [Random Forest](#random-forest)
  - [Gaussian Naive Bayes](#gaussian-naive-bayes)
  - [K-Nearest Neighbors](#k-nearest-neighbors)
- [Random Forest](#random-forest)
  - [Data Standardization](#data-standardization)
  - [Training and Testing](#training-and-testing)
- [Gaussian Naive Bayes](#gaussian-naive-bayes)
- [K-Nearest Neighbors](#k-nearest-neighbors)
  - [Nearest Neighbor Categorization](#nearest-neighbor-categorization)
  - [Data Preprocessing](#data-preprocessing-1)
  - [Training and Testing](#training-and-testing-1)
  - [Classification Metrics](#classification-metrics)
  - [ROC Graph](#roc-graph)

## Introduction

The examined group comprised kernels belonging to three different varieties of wheat: Kama, Rosa, and Canadian, 70 elements each, randomly selected for the experiment. High-quality visualization of the internal kernel structure was detected using a soft X-ray technique. It is non-destructive and considerably cheaper than other more sophisticated imaging techniques like scanning microscopy or laser technology. The images were recorded on 13x18 cm X-ray KODAK plates. Studies were conducted using combine-harvested wheat grain originating from experimental fields, explored at the Institute of Agrophysics of the Polish Academy of Sciences in Lublin. The data set can be used for the tasks of classification and cluster analysis.

## Data Preprocessing

### Data Import
First, we'll need to import the libraries we'll need for this project. Next, we'll need to find the amount of seeds that correspond to each type of fruit out of 3. The 3 types of wheat varieties are Karma, Rosa, and Canadian, 70 items each, randomly selected for the experiment.

### Data Representation
Once this is done, there are two equally important steps we'll take that will help us a lot in understanding our data. The `info()` and `describe()` functions are used. The `info` function gives as a result the type of data that exists in each of the data columns. The `describe` function is used to display some basic statistical details, such as percentage, average, std, etc., of a data frame or array of numeric values.

### Correlation Analysis
Another step for further understanding and analysis of our data is representation. We use `df['seedType']` to access the `seedType` field and call `value_counts` to get a set of unique values. We normalize these values. Data normalization is used to make model training less sensitive to the scale of features. This allows our model to converge better and leads to a more accurate model.

## Correlation Heatmap

The next step will be to calculate the correlation coefficient. One way to find the correlation between two variables is to create a scatter plot where we represent pairs of observations. If we perform the specific procedure for all combinations of columns (variables), we can graphically represent their correlation. In this way, we have the ability to graphically observe the correlation between two variables.

We will create diagrams for this purpose with the help of the seaborn library. Seaborn is a Python data visualization library based on matplotlib. More specifically, we will use `PairGrid`. This object maps each variable to a dataset in a column and a row in a multi-axis grid. But since we want to represent a lot of data, we will use the `pairplot()` method, which serves the representation of many graphs in one line.

## Data Pre-processing

Next, the process is followed by the pre-processing of the data. For data preparation, one of the most useful transformations is data scaling. This process converts the data in such a way that the values of the different columns (variables) change within a certain range. This reduces the effect of the difference in the numerical scales of the data (e.g., one variable can vary between 0-1 and another between -1000 and 10000000) on the machine learning algorithms. We import the `StandardScaler()` function from scikit-learn. The data standardization method will be used.

## Train and Test Sets

Then we continue to create training and control sets. The purpose is to train the algorithm to respond correctly to future data (efficiency in making correct predictions). We will use the `train_test_split()` function of scikit-learn. Essentially, it splits the tables into random train and test subsets. The training set is a subset to 'train' the model, and the test set is a subset to test the trained model.

## Classification Algorithms

Next, we use 3 algorithms for categorization. The algorithms used are as follows:

### Random Forest

Firstly, we'll analyze the first algorithm, that of the random forest. A random forest is an estimator that fits a number of tree decision classifiers into different subsets of the data set and uses the average to improve forecast accuracy and over-alignment control. The random forest, as its name suggests, consists of a large number of individual decision trees that function as a whole. Each individual tree in the random forest unfolds a class prediction, and the class with the most votes becomes our model prediction.

#### Data Standardization

In the code, we import the `StandardScaler()` function from the sklearn.model_selection library. The StandardScaler function will transform the data so that the distribution has an average value of 0 and a standard deviation of 1. Then we adjust the data so that we can then convert it. We train the data and test them and then continue with the random forest.

#### Training and Testing

We import the appropriate libraries and use the `RandomForestClassifier` function to define the evaluators. We make the forest with the trees from the set with the trained variables and then with the functions `predict()` and `predict_proba()` we calculate the order and the probability classes of the trained x. The same procedure is repeated for test_X. We print for both the confusion table as well as the accuracy score. The confusion table evaluates the output quality of a classifier. The diagonal elements represent the number of points for which the predicted label is equal to the actual label, while the other elements outside the main diagonal are those that have not been correctly marked by the classifier. The higher the diagonal values of the confusion table, the more accurate predictions we have.

#### Gaussian Naive Bayes

Next is the Gaussian Naive Bayes algorithm for categorization. Bayesian naive classifiers are a family of simple "probabilistic classifiers" based on the application of Bayes' theorem with strong (naive) assumptions of independence between attributes. In general, Naive Bayes methods are a set of supervised learning algorithms based on the application of Bayes' theorem with the "naive" assumption of conditional independence between each pair of attributes given the value of the variable class. The naive Bayes classifiers work quite well in many real-world situations, such as document sorting and spam filtering. They require a small amount of training data to assess the necessary parameters. Naive Bayes learners and classifiers are quite fast compared to other methods. Each distribution can be independently estimated as a one-dimensional distribution. This helps to solve the problems that arise due to the spaces. Nevertheless, the naive Bayes may be good classifiers but as appraisers they are not as good. 

In the code, we start once again with the training and testing of the data after we have first introduced from the sklearn.naive_bayes library the GaussianNB equation (which is used for classification in Naive Bayes). Then we find with the functions `predict()` and `predict_proba()` the forecasts for the training and testing sets.

#### K-Nearest Neighbors

At last, we have the nearest neighbor algorithm. The nearest neighbor categorization is part of a more general technique, known as snapshot training, which does not construct a general model, but uses specific instructional samples to make predictions for a control snapshot. Such algorithms require a proximity measure to determine the similarity or distance between the snapshots and a categorization function, which returns the predicted category of a control snapshot based on its proximity to other snapshots.

In the code, we first import the appropriate libraries and then find the nearest neighbors of a point. We specify the parameters and create a variable to find the nearest neighbors and then print them. With the `fit()` function, we adapt the categorizer of the nearest neighbors to the elements we have trained. After that, we use once again the functions `predict()` and `predict_proba()` to calculate the order and the probability classes of the trained x. The same procedure is repeated for test_X. We print for both the confusion table as well as the accuracy score.

#### Classification Metrics

As the proper pre-processing and training of the data has been done, the algorithm is quite reliable. As shown in the picture based on the confusion table for both train and test data, the diagonal consists of quite large numbers, which indicates that we have many correct predictions.

Based on the validity score for the train data, we have a validity of 0.91 (i.e., 91%), which means that we have valid results. For the test data, we have a validity score of 0.93 (i.e., 93%), which is an equally high validity rate.

Then I print the main categorization metrics for this algorithm using the `classification_report()` function.

- Precision is the ability of the categorizer not to label as positive a sample that is negative.
- Recall is the ability of the categorizer to find all the positive samples.
- F1-score is an average of accuracy and recall (precision & recall). The best value reaches 1 and the worst at 0.
- Support is the number of occurrences of each `y_true` class.
- Accuracy calculates the accuracy of the subset: the set of labels provided for a sample must exactly match the corresponding set of labels in `y_true`.
- Macro avg calculates the measurements for each tag and finds the unweighted average.
- Weighted avg calculates the metrics for each tag and finds their average weight based on support.

#### ROC Graph

Finally, I print the ROC graph, which usually has a true positive rate on the Y-axis and a false positive rate on the X-axis. This means that the upper left corner of the graph is the "ideal" point - a false positive percentage zero, and a real positive percentage one. This is not very realistic, but it does mean that a larger area under the curve (AUC) is usually better. The "deviation" of the ROC curves is also important, as it is ideal to maximize the true positive percentage while minimizing the false positive percentage.

