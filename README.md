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

Firstly, we'll analyze the first algorithm, that of the random forest. A random forest is an estimator that fits a number of tree decision classifiers into different subsets of the data set and uses the average to improve forecast accuracy and over-alignment control. The random forest, as its name suggests, consists of a large number of individual decision trees that function as a whole
