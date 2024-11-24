## Table of Contents
- [Basic Overview](#basic-overview)
- [Regression in Detail](#regression-in-detail)
- [Classification in Detail](#classification-in-detail)
- [Clustering in Detail](#clustering-in-detail)

## Basic Overview

### Regression Algorithms 

*   **Linear Regression:**
    *   **Problem:** Predicts a continuous target variable based on a linear relationship with one or more predictor variables.
    *   **Formula:**  f(X) = β + ω<sup>T</sup>X, where f(X) is the predicted value, β is the bias term, ω is the vector of regression coefficients, and X is the vector of predictor variables.
    *   **Key Concept:** Minimises the sum of squared errors (SSE) between predicted and actual values to find the best-fit line.
*   **Multilinear Regression:** 
    *   **Problem:** Extends linear regression to handle multiple predictor variables. Also known as Multiple Regression in the sources. 
    *   **Formula:**  f(X) = β + Σ<sub>(i=1 to d)</sub> ω<sub>i</sub>X<sub>i</sub>, where *d* is the number of predictor variables.
    *   **Key Concept:** Uses techniques like QR-factorization and Gram-Schmidt orthogonalisation to find the optimal regression coefficients.
*   **Polynomial Regression:**
    *   **Problem:** Models non-linear relationships between predictor and target variables using polynomial functions. 
    *   **Formula:**  f(X) = β + Σ<sub>(i=1 to d)</sub>Σ<sub>(j=1 to p)</sub> ω<sub>ij</sub>X<sub>i</sub><sup>j</sup>, where *p* is the degree of the polynomial. 
    *   **Key Concept:** Not explicitly covered in the sources, but achievable through kernel regression with a polynomial kernel.
*   **Lasso Regression:**
    *   **Problem:** Performs both variable selection and regularisation to prevent overfitting in linear regression models. Also known as L1 regression in the sources. 
    *   **Formula:** Minimises the objective function: 1/2 \* ||Y-Dw||<sup>2</sup> + α||w||<sub>1</sub>, where Y is the response vector, D is the data matrix, w is the weight vector, and α controls the amount of regularisation.
    *   **Key Concept:** Uses a penalty term based on the L1 norm of the regression coefficients, forcing some coefficients to become zero, effectively selecting a subset of important features.
*   **Ridge Regression:**
    *   **Problem:** Addresses multicollinearity (high correlation between predictor variables) in linear regression by shrinking the regression coefficients. Also known as L2 regression in the sources. 
    *   **Formula:** Minimises the objective function: 1/2 \* ||Y-Dw||<sup>2</sup> + α/2 \* ||w||<sub>2</sub><sup>2</sup>, where α controls the strength of the penalty.
    *   **Key Concept:** Adds a penalty term based on the L2 norm of the coefficients to the least squares objective function, preventing coefficients from becoming too large and reducing the model's sensitivity to individual data points.

### Classification Algorithms 

*   **Logistic Regression:**
    *   **Problem:** Predicts the probability of a binary outcome (0 or 1) based on a set of predictor variables. 
    *   **Formula:**  P(Y=1|X) = 1 / (1+e<sup>-(β + ω<sup>T</sup>X)</sup>), where P(Y=1|X) is the probability of Y=1 given X.
    *   **Key Concept:** Uses a sigmoid function to transform a linear combination of predictor variables into a probability.
*   **Decision Tree:**
    *   **Problem:** Creates a tree-like model to classify data points by recursively partitioning the feature space based on the values of input features.
    *   **Formula:** Several formulae are involved, including entropy, information gain, Gini index, and CART for evaluating split points.
    *   **Key Concept:** Aims to create pure leaf nodes where data points mostly belong to the same class. Can handle both numerical and categorical data. 
*   **Support Vector Machine:**
    *   **Problem:** Finds the optimal hyperplane that maximises the margin between two classes. 
    *   **Formula:** The hyperplane is defined as: h(x) = w<sup>T</sup>x + b = 0, where *w* is the weight vector, *x* is the input vector, and *b* is the bias.
    *   **Key Concept:** The support vectors are the data points closest to the hyperplane and are critical in defining the decision boundary. Can handle non-linear classification using the kernel trick.
*   **Naive Bayes:**
    *   **Problem:** A probabilistic classifier that applies Bayes' theorem with a strong (naive) independence assumption between features.
    *   **Formula:**  P(c<sub>i</sub>|x) = (P(x|c<sub>i</sub>) \* P(c<sub>i</sub>)) / P(x), where P(c<sub>i</sub>|x) is the posterior probability of class c<sub>i</sub> given the feature vector *x*.
    *   **Key Concept:** Assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. Despite its simplicity, it can be surprisingly effective. 
*   **Gaussian Bayes:**
    *   **Problem:** A specific type of Naive Bayes classifier where the likelihood of each feature belonging to a class is modelled using a Gaussian (normal) distribution.
    *   **Formula:**  P(x<sub>j</sub>|c<sub>i</sub>) = (1 / √(2πσ<sub>ij</sub><sup>2</sup>)) \* exp(-(x<sub>j</sub> - μ<sub>ij</sub>)<sup>2</sup> / 2σ<sub>ij</sub><sup>2</sup>), where x<sub>j</sub> is the jth feature, μ<sub>ij</sub> and σ<sub>ij</sub> are the mean and standard deviation of the jth feature for class c<sub>i</sub>.
    *   **Key Concept:** Often used when dealing with continuous data, assuming that the data for each class follows a normal distribution.
*   **Random Forest:**
    *   **Problem:** An ensemble learning method that combines multiple decision trees to improve prediction accuracy and generalisation.
    *   **Formula:** No specific formula, but it relies on the concepts of decision trees and ensemble learning by averaging or voting on the predictions of multiple trees.
    *   **Key Concept:** Constructs each decision tree using a random subset of features and data points. This randomness helps to decorrelate the trees and reduce overfitting.

### Clustering Algorithms

*   **K-means:**
    *   **Problem:** Partitions data points into *k* clusters where each point belongs to the cluster with the nearest mean (centroid).
    *   **Formula:**  Minimises the sum of squared errors (SSE): SSE(C) = Σ<sub>(i=1 to k)</sub>Σ<sub>(xj ∈ Ci)</sub> ||x<sub>j</sub> - μ<sub>i</sub>||<sup>2</sup>, where C<sub>i</sub> represents the ith cluster and μ<sub>i</sub> is its centroid.
    *   **Key Concept:**  An iterative algorithm that alternates between assigning points to clusters and updating cluster centroids. Can be extended to handle non-linearly separable data using kernel k-means.


## Regression in Detail

### Overview
Regression analysis is a statistical method that helps us understand and quantify the relationship between a dependent variable and one or more independent variables. It aims to find a function that best describes this relationship, allowing us to predict the value of the dependent variable based on the values of the independent variables. In machine learning, regression algorithms are widely used for tasks like:

* Predicting stock prices
* Forecasting sales
* Estimating house prices
* Analyzing the effectiveness of marketing campaigns

### Working
#### Linear Regression
Linear regression assumes a linear relationship between the dependent variable (Y) and the independent variable (X). The goal is to find the best-fitting line that minimizes the sum of squared errors (SSE) between the predicted values and actual values.

**Formula:**

ŷ = b + w * x

Where:
* ŷ: The predicted value of the dependent variable.
* b: The intercept of the line (bias term).
* w: The slope of the line (regression coefficient).
* x: The value of the independent variable.

**Applications:**
* Predicting sales based on advertising spending.
* Estimating crop yield based on rainfall.

#### Multilinear Regression
Multilinear regression extends linear regression to cases with multiple independent variables (X<sub>1</sub>, X<sub>2</sub>, ..., X<sub>d</sub>).

**Formula:**

ŷ = b + w<sub>1</sub> * x<sub>1</sub> + w<sub>2</sub> * x<sub>2</sub> + ... + w<sub>d</sub> * x<sub>d</sub>

Where:
* ŷ: The predicted value of the dependent variable.
* b: The intercept of the hyperplane (bias term).
* w<sub>i</sub>: The regression coefficient for the i-th independent variable.
* x<sub>i</sub>: The value of the i-th independent variable.

**Applications:**
* Predicting house prices based on factors like size, location, and number of bedrooms.
* Analyzing the impact of multiple factors on a company's stock price.

#### Polynomial Regression
Polynomial regression models non-linear relationships between variables by using polynomial functions of the independent variables.

**Formula:**

ŷ = b + w<sub>1</sub> * x + w<sub>2</sub> * x<sup>2</sup> + ... + w<sub>d</sub> * x<sup>d</sup>

Where:
* ŷ: The predicted value of the dependent variable.
* b: The intercept (bias term).
* w<sub>i</sub>: The regression coefficient for the i-th term.
* x: The value of the independent variable.
* d: The degree of the polynomial.

**Applications:**
* Modeling the growth of populations.
* Predicting the trajectory of a projectile.

#### Lasso Regression
Lasso regression adds an L1 penalty to the least squares objective function, encouraging sparsity in the weight vector by shrinking some coefficients to zero. This acts as an automatic feature selection mechanism.

**Formula:**

Minimize: 1/2 * ‖Y - Dw‖<sup>2</sup> + α * ‖w‖<sub>1</sub>

Where:
* Y: The response vector.
* D: The data matrix.
* w: The weight vector.
* α: The regularization parameter controlling the strength of the L1 penalty.

**Applications:**
* Feature selection in high-dimensional datasets.
* Building simpler and more interpretable models.

#### Ridge Regression
Ridge regression adds an L2 penalty to the least squares objective function, shrinking the regression coefficients to prevent overfitting. Unlike Lasso, Ridge does not force coefficients to zero.

**Formula:**

Minimize: 1/2 * ‖Y - Dw̃‖<sup>2</sup> + α * ‖w̃‖<sup>2</sup><sub>2</sub>

Where:
* Y: The response vector.
* D: The data matrix.
* w̃: The augmented weight vector (including bias).
* α: The regularization parameter controlling the strength of the L2 penalty.

**Applications:**
* Dealing with multicollinearity (high correlation between independent variables).
* Improving model generalization by reducing overfitting.

#### Support Vector Regression (SVR)
Support Vector Regression (SVR) uses support vector machines to perform regression tasks. It aims to find a function that deviates from the actual data points by a maximum margin ε (epsilon), tolerating errors within this margin.

**Formula:**

Minimize: 1/2 * ‖w‖<sup>2</sup> + C * Σ(ξ<sub>i</sub> + ξ<sub>i</sub>*)

Where:
* w: The weight vector.
* C: A regularization parameter controlling the trade-off between the flatness of the function and the amount up to which deviations larger than ε are tolerated.
* ξ<sub>i</sub>, ξ<sub>i</sub>*: Slack variables representing the deviations above and below ε.
* ε: The maximum margin of error allowed.

**Applications:**
* Predicting time series data.
* Modeling complex non-linear relationships.

### Basic Concepts
#### Assumptions for Regression Models
* **Linearity:** The relationship between the independent and dependent variables is assumed to be linear (or can be transformed to be linear).
* **Independence:** The errors (residuals) are assumed to be independent of each other.
* **Homoscedasticity:** The errors have constant variance across different values of the independent variables.
* **Normality:** The errors are assumed to be normally distributed.

#### Overfitting and Underfitting
* **Overfitting:** The model learns the training data too well, capturing noise and random fluctuations, and performs poorly on unseen data.
* **Underfitting:** The model is too simple and fails to capture the underlying patterns in the data, leading to poor performance on both training and unseen data.

**Causes and Solutions:**
* **Overfitting:**
    * **Causes:** Too many features, complex model, insufficient data.
    * **Solutions:** Regularization, feature selection, cross-validation, more data.
* **Underfitting:**
    * **Causes:** Too few features, overly simple model.
    * **Solutions:** Adding more features, using a more complex model.

#### Evaluation Metrics
* **R-squared:** Measures the proportion of variance in the dependent variable explained by the model. A higher R-squared indicates a better fit.
* **Mean Squared Error (MSE):** Averages the squared errors between predicted and actual values. A lower MSE indicates a better fit.

#### Regularization
* **L1 norm (Lasso):** Sum of the absolute values of the coefficients. Induces sparsity by shrinking some coefficients to zero.
* **L2 norm (Ridge):** Sum of the squared values of the coefficients. Shrinks coefficients towards zero but does not force them to be exactly zero.

#### The Role of Hyperparameters
* **Alpha (α) for Lasso/Ridge:** Controls the strength of regularization. Higher alpha leads to more shrinkage of coefficients.
* **Epsilon (ε) for SVR:** Defines the margin of tolerance around the regression function.

#### Handling Multicollinearity and Feature Selection
* **Multicollinearity:** Occurs when independent variables are highly correlated, making it difficult to isolate the individual effects of each variable. 
    * **Solution:** Feature selection, dimensionality reduction techniques (e.g., principal component analysis), regularization (Ridge regression).
* **Feature selection:** Selecting a subset of relevant features can improve model performance, reduce overfitting, and enhance interpretability.
    * **Methods:** Lasso regression, forward/backward selection, recursive feature elimination.

**Note:** This response only includes information from the provided sources. It does not contain any information from outside sources.


## Classification in Detail

### Overview
Classification is a supervised learning technique where the goal is to categorise data points into predefined classes. It plays a crucial role in machine learning, enabling us to:

* Filter spam emails.
* Detect fraudulent transactions.
* Diagnose diseases based on symptoms.
* Recognise objects in images.

### Working

#### Logistic Regression
Logistic regression, despite its name, is a classification algorithm. It predicts the probability of a data point belonging to a specific class using the sigmoid function.

**Sigmoid Function:**

P(y=1|x) = 1 / (1 + exp(-(b + w<sup>T</sup>x)))


Where:
* P(y=1|x): Probability of the data point x belonging to class 1.
* b: The intercept (bias term).
* w: The weight vector.
* x: The data point.

**Applications:**
* Credit scoring.
* Customer churn prediction.

**Pseudo-code:**

1. Initialise the weight vector and bias.
2. For each data point:
   * Calculate the predicted probability using the sigmoid function.
   * Update the weights and bias based on the prediction error.
3. Repeat step 2 until convergence.

**Python Code:**
```
python
from sklearn.linear_model import LogisticRegression

# Create a logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)
```

#### Decision Tree
A decision tree uses a tree-like structure to make decisions, splitting the data based on feature values until a leaf node representing a class label is reached.

**Applications:**
* Medical diagnosis.
* Customer segmentation.

**Pseudo-code:**

1. Start with the entire dataset.
2. Select the best feature to split the data based on an impurity measure (e.g., Gini index).
3. Create child nodes representing the split data.
4. Recursively repeat steps 2-3 for each child node until a stopping condition is met (e.g., maximum depth, minimum leaf size).
5. Assign a class label to each leaf node based on the majority class in that node.

**Python Code:**
```
python
from sklearn.tree import DecisionTreeClassifier

# Create a decision tree model
model = DecisionTreeClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)
```

#### Support Vector Machine (SVM)
SVM aims to find the optimal hyperplane that maximises the margin between different classes. It can handle linear and non-linear classification using kernels.

**Formula for linear SVM:**


h(x) = w<sup>T</sup>x + b


Where:
* h(x): The hyperplane function.
* w: The weight vector.
* b: The bias term.
* x: The data point.

**Applications:**
* Image classification.
* Text categorisation.

**Pseudo-code:**

1. For each data point:
   * Calculate the distance to the hyperplane.
   * Identify support vectors (points closest to the hyperplane).
2. Optimise the hyperplane to maximise the margin while minimising misclassifications.

**Python Code:**
```
python
from sklearn.svm import SVC

# Create an SVM model
model = SVC(kernel='linear')  # Use a linear kernel

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)
```

#### Naive Bayes
Naive Bayes applies Bayes' theorem with the naive assumption of independence between features. Despite its simplicity, it can be surprisingly effective.

**Bayes' Theorem:**


P(c<sub>i</sub>|x) = (P(x|c<sub>i</sub>) * P(c<sub>i</sub>)) / P(x)


Where:
* P(c<sub>i</sub>|x): Posterior probability of class c<sub>i</sub> given data point x.
* P(x|c<sub>i</sub>): Likelihood of data point x given class c<sub>i</sub>.
* P(c<sub>i</sub>): Prior probability of class c<sub>i</sub>.
* P(x): Probability of data point x.

**Applications:**
* Spam filtering.
* Sentiment analysis.

**Pseudo-code:**

1. Calculate the prior probabilities of each class.
2. For each class:
   * Calculate the likelihood of each feature value given the class.
3. For a new data point:
   * Calculate the posterior probability of each class using Bayes' theorem.
   * Predict the class with the highest posterior probability.

**Python Code:**
```
python
from sklearn.naive_bayes import GaussianNB

# Create a Gaussian Naive Bayes model
model = GaussianNB()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

```
#### Gaussian Bayes
Gaussian Bayes is a specific type of Naive Bayes where the likelihood of each feature is modelled using a Gaussian (normal) distribution.

**Applications:**
* Similar to Naive Bayes, but specifically useful when features are continuous and assumed to follow a normal distribution.

#### Random Forest
Random forest combines multiple decision trees trained on different subsets of data and features to improve accuracy and robustness.

**Applications:**
* Object detection.
* Medical diagnosis.

**Pseudo-code:**

1. Create multiple bootstrap samples from the training data.
2. For each bootstrap sample:
   * Train a decision tree, randomly selecting a subset of features at each node.
3. For a new data point:
   * Get predictions from all trees.
   * Predict the class by majority voting among the trees.

**Python Code:**
```
python
from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest model
model = RandomForestClassifier(n_estimators=100)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)
```

### Basic Concepts

#### Types of Classification Problems

* **Binary Classification:** Predicting one of two possible classes (e.g., spam or not spam).
* **Multiclass Classification:** Predicting one of multiple possible classes (e.g., classifying different types of flowers).

#### Key Evaluation Metrics

* **Precision:** The proportion of correctly predicted positive instances among all instances predicted as positive.
* **Recall:** The proportion of correctly predicted positive instances among all actual positive instances.
* **F1-Score:** The harmonic mean of precision and recall, providing a balanced measure of accuracy.
* **ROC-AUC:** The area under the Receiver Operating Characteristic (ROC) curve, which plots the true positive rate against the false positive rate. A higher AUC indicates better performance.

#### Decision Boundaries

* **Linear Classification:** Separating classes using a straight line (in 2D) or hyperplane (in higher dimensions).
* **Non-linear Classification:** Separating classes using non-linear boundaries (curves, surfaces, etc.).

#### Overfitting and Underfitting

* **Overfitting:** Model performs well on training data but poorly on unseen data.
   * **Solutions:** Regularisation (e.g., in logistic regression, SVM), pruning in decision trees, more data.
* **Underfitting:** Model performs poorly on both training and unseen data.
   * **Solutions:** More complex model, adding more features.

#### Handling Imbalanced Datasets

* **Oversampling (SMOTE):** Generating synthetic data points for the minority class to balance the dataset.
* **Undersampling:** Removing data points from the majority class to balance the dataset.
* **Weighted Classes:** Assigning different weights to different classes during training to account for imbalance.

#### Hyperparameter Tuning

* **SVM:**
   * **Kernel:** Determines the type of decision boundary (linear, polynomial, RBF).
   * **C:** Controls the trade-off between the margin width and misclassifications.
   * **Gamma:** Influences the shape of the decision boundary for non-linear kernels.
* **Random Forest:**
   * **Number of trees:** More trees generally improve performance but increase computational cost.
   * **Maximum depth:** Limits the depth of each tree to prevent overfitting.

#### Comparison of Naive Bayes vs. Gaussian Bayes

* Naive Bayes assumes independence between features, while Gaussian Bayes further assumes that features follow a Gaussian distribution.
* Gaussian Bayes is more suitable when features are continuous and normally distributed.
* Naive Bayes can handle categorical features as well.

**Note:** This response only includes information from the provided sources. It does not contain any information from outside sources.



## Clustering in Detail

### Overview
Clustering is an unsupervised learning technique that groups data points into clusters based on similarity. It is widely used in machine learning for tasks like:

* **Customer segmentation:** Grouping customers with similar purchasing behaviour.
* **Anomaly detection:** Identifying outliers that deviate significantly from normal patterns. 
* **Image compression:** Reducing image size by grouping similar pixels.

### K-Means Clustering
#### Working
**K-Means** is a representative-based clustering algorithm that partitions data into *k* clusters, where each cluster is represented by its **centroid** (the mean of all points in the cluster).

**Steps of the K-Means Algorithm:**
1. **Initialisation:** Randomly select *k* points in the data space as initial centroids.
2. **Assignment:** Assign each data point to the cluster whose centroid is closest.
3. **Update:** Recalculate the centroid of each cluster based on the newly assigned points.
4. **Repeat steps 2 and 3** until convergence (centroids no longer change significantly or a fixed number of iterations is reached).

#### Formulae
* **Sum of Squared Errors (SSE):** This function measures the quality of clustering by calculating the sum of squared distances between each point and its cluster centroid.


SSE(C) = ∑<sub>i=1</sub><sup>k</sup> ∑<sub>x<sub>j</sub>∈C<sub>i</sub></sub> ||x<sub>j</sub> - µ<sub>i</sub>||<sup>2</sup> 


Where:

    * C: The clustering (set of all clusters).
    * C<sub>i</sub>: The ith cluster.
    * x<sub>j</sub>: The jth data point in cluster C<sub>i</sub>.
    * µ<sub>i</sub>: The centroid of cluster C<sub>i</sub>.
    * ||x<sub>j</sub> - µ<sub>i</sub>||: The Euclidean distance between data point x<sub>j</sub> and centroid µ<sub>i</sub>.

#### Applications
* **Market Segmentation:** Grouping customers based on purchasing habits, demographics, or other factors to tailor marketing strategies.
* **Image Segmentation:** Partitioning an image into regions with similar visual features, used in object recognition and image editing.
* **Anomaly Detection:** Identifying data points that deviate significantly from the norm, potentially indicating fraud or system errors.

#### Pseudo-code

function K-MEANS(D, k, ϵ):
    t = 0
    Randomly initialise k centroids: µ<sub>1</sub><sup>t</sup>, µ<sub>2</sub><sup>t</sup>, ..., µ<sub>k</sub><sup>t</sup> ∈ R<sup>d</sup>
    
    repeat
        t ← t + 1
        C<sub>i</sub> ← ∅ for all i = 1, ..., k
        
        // Cluster Assignment Step
        for each x<sub>j</sub> ∈ D do
            i* ← argmin<sub>i</sub>(||x<sub>j</sub> - µ<sub>i</sub><sup>t-1</sup>||<sup>2</sup>)
            C<sub>i*</sub> ← C<sub>i*</sub> ∪ {x<sub>j</sub>}  
        
        // Centroid Update Step
        for each i = 1, ..., k do
            µ<sub>i</sub><sup>t</sup> ← (1 / |C<sub>i</sub>|) ∑<sub>x<sub>j</sub>∈C<sub>i</sub></sub> x<sub>j</sub>
    
    until ∑<sub>i=1</sub><sup>k</sup> ||µ<sub>i</sub><sup>t</sup> - µ<sub>i</sub><sup>t-1</sup>||<sup>2</sup> ≤ ϵ
    
    return C

### Basic Concepts
#### Key Terms
* **Centroid:** The mean of all data points in a cluster.
* **Inertia:** The sum of squared distances of samples to their closest cluster center, similar to SSE.
* **Cluster:** A group of data points considered similar based on a distance metric.

#### Steps of the K-Means Algorithm
* **Initialisation:** Choosing *k* initial cluster centroids.
* **Assignment:** Assigning each data point to the closest centroid.
* **Update:** Recomputing the centroids based on the assigned points.

#### Evaluation Metrics for Clustering
* **Silhouette Score:** Measures how similar an object is to its own cluster compared to other clusters. Higher values indicate better clustering.
* **Elbow Method:** Plots the SSE against the number of clusters (*k*). The "elbow" point suggests a good value for *k*.
* **Davies-Bouldin Index:** Measures the average similarity between each cluster and its most similar cluster. Lower values indicate better clustering. 

### Limitations of K-Means
* **Sensitivity to Initialization:** Results can vary depending on the initial centroid positions.
* **Sensitivity to Outliers:** Outliers can significantly distort the cluster centroids.
* **Assumes Spherical Clusters:** May not perform well on data with complex, non-spherical cluster shapes.

### Enhancements to K-Means
* **K-Means++:** Uses a smarter initialisation strategy to mitigate sensitivity to initial centroid placement.
* **Kernel K-Means:** Applies the kernel trick to handle non-linearly separable clusters.
* **Soft K-Means:** Allows partial cluster membership, assigning probabilities of belonging to multiple clusters.

### Choosing the Optimal Number of Clusters (k)
* **Elbow Method:** Look for the "elbow" in the plot of SSE vs. *k*. 
* **Silhouette Analysis:** Choose *k* that maximises the Silhouette Score.

### Python Code Examples (scikit-learn)
```
python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Create a KMeans model with k=4
kmeans = KMeans(n_clusters=4, random_state=42)

# Fit the model to the data
kmeans.fit(X)

# Get the cluster labels
labels = kmeans.labels_

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.show()

# Evaluate using the Silhouette Score
silhouette_avg = silhouette_score(X, labels)
print(f"Silhouette Score: {silhouette_avg}")

```
**Explanation:**

* **Import necessary libraries.**
* **Generate sample data** using `make_blobs`.
* **Create a KMeans model** with the desired number of clusters.
* **Fit the model to the data** using `kmeans.fit(X)`.
* **Obtain cluster labels** for each data point.
* **Visualize the clusters** using `matplotlib`.
* **Evaluate the clustering** using the Silhouette Score. 
