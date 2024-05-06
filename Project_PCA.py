#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12,8)

 # data URL: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                  header=None)

iris.head()


# In[4]:


iris.columns = ["sepal_length", "sepal_width", "petal_length","petal_width", "species"] 
iris.dropna(how='all', inplace = True)
iris.head()


# In[5]:


iris.info()


# In[6]:


sns.scatterplot(x = iris.sepal_length, y = iris.sepal_width, 
                hue = iris.species,
                style = iris.species);


# In[7]:


X = iris.iloc[:, 0:4].values
y = iris.species.values

from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)


# In[9]:


covariance_matrix = np.cov(X.T)
print("covariance matrix: \n", covariance_matrix)


# In[10]:


eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
print("eigenvalues: :\n", eigen_values, "\n eigenvectors: \n", eigen_vectors)


# In[11]:


eigen_vec_svd, s, v = np.linalg.svd(X.T)
eigen_vec_svd


# In[12]:


for val in eigen_values :
    print(val)


# In[13]:


variance_explained = [(i/sum(eigen_values))*100 for i in eigen_values]
variance_explained


# In[14]:


cumulative_variance_explaiend = np.cumsum(variance_explained)
cumulative_variance_explaiend


# In[15]:


sns.lineplot(x= [1, 2, 3, 4], y = cumulative_variance_explaiend);
plt.xlabel("Number of components")
plt.ylabel("cumulative explaiend variance ")
plt.title("Explaiend variance vs Number of compenents")
plt.show()


# In[17]:


eigen_vectors


# In[18]:


projection_matrix = (eigen_vectors.T[:][:])[:2].T
print("projection matrix: \n", projection_matrix)


# In[55]:


X_pca = X.dot(projection_matrix)

species_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
species_colors = ['blue', 'orange', 'green']


for i, species in enumerate(species_labels):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=species, color=species_colors[i], alpha=0.7)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.legend()
plt.show()



# In[57]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset from the provided URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris = pd.read_csv(url, header=None, names=column_names)

# Drop any rows with missing values
iris.dropna(inplace=True)

# Select numerical columns for correlation analysis
numerical_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Calculate the correlation matrix
corr_matrix = iris[numerical_columns].corr()

# Plotting the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True, fmt='.2f',
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Iris Dataset Features')
plt.show()


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                   header=None)

# Naming columns
iris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"] 

# Drop any missing values
iris.dropna(how='all', inplace=True)

# Display the info
iris.info()

# Set figure size
plt.figure(figsize=(12, 8))

# Create boxplot using seaborn
sns.boxplot(data=iris.drop('species', axis=1))  # Exclude species column from boxplot

# Customize plot labels and title
plt.xlabel('Features')
plt.ylabel('Measurement (cm)')
plt.title('Boxplot of Iris Dataset Features')

# Display the plot
plt.show()


# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the Iris dataset
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                   header=None)

# Naming columns
iris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"] 

# Drop any missing values
iris.dropna(how='all', inplace=True)

# Set figure size
plt.figure(figsize=(12, 8))

# Create pair plot using seaborn
sns.pairplot(iris, hue="species", markers=["o", "s", "D"])

# Customize plot title
plt.suptitle("Pair Plot of Iris Dataset by Species", y=1.02)

# Display the plot
plt.show()


# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                   header=None)
iris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris.dropna(how='all', inplace=True)

# Standardize the features
X = StandardScaler().fit_transform(iris.iloc[:, 0:4].values)

# Compute covariance matrix
covariance_matrix = np.cov(X.T)

# Compute eigenvalues and eigenvectors
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

# Calculate explained variance ratios
variance_explained = (eigen_values / np.sum(eigen_values)) * 100

# Sort the eigenvalues and variance explained in descending order
sorted_indices = np.argsort(eigen_values)[::-1]
sorted_variance_explained = variance_explained[sorted_indices]

# Calculate cumulative explained variance
cumulative_variance_explained = np.cumsum(sorted_variance_explained)

# Create a Pareto plot
plt.figure(figsize=(14, 10))

# Plot bars for individual explained variance
sns.barplot(x=np.arange(1, 5), y=sorted_variance_explained, color='skyblue', alpha=0.7)

# Plot cumulative explained variance line
plt.plot(np.arange(0, 4), cumulative_variance_explained, marker='o', color='orange', label='Cumulative Explained Variance')

# Add annotations to bars and cumulative line
for i, (var, cum_var) in enumerate(zip(sorted_variance_explained, cumulative_variance_explained)):
    plt.text(i + 0, var + 1, f'{var:.1f}%', fontsize=10, color='black', fontweight='bold')
    plt.text(i + 0, cum_var - 4, f'{cum_var:.1f}%', fontsize=10, color='orange', fontweight='bold')

# Highlight key components contributing to most of the variance (e.g., > 80%)
plt.axhline(y=80, color='red', linestyle='--', label='80% Explained Variance')
plt.legend()

# Customize plot title and labels
plt.title('Pareto Plot of Explained Variance Ratios by PCA Components')
plt.xlabel('PCA Components')
plt.ylabel('Explained Variance (%)')

# Show the plot
plt.show()


# In[30]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                   header=None)
iris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris.dropna(how='all', inplace=True)

# Standardize the features
X = StandardScaler().fit_transform(iris.iloc[:, 0:4].values)

# Compute covariance matrix
covariance_matrix = np.cov(X.T)

# Compute eigenvalues and eigenvectors
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

# Sort eigenvalues and corresponding eigenvectors by descending order of eigenvalues
sorted_indices = np.argsort(eigen_values)[::-1]
sorted_eigen_vectors = eigen_vectors[:, sorted_indices]

# Summarize coefficients across all principal components
total_coefficients = np.abs(sorted_eigen_vectors).sum(axis=1)

# Create a scatter plot of total coefficient values for each feature
plt.figure(figsize=(10, 6))
features = iris.columns[:-1]
plt.scatter(features, total_coefficients, color='skyblue', alpha=0.7, s=100, edgecolors='black')
plt.title('Total Coefficients Across Principal Components')
plt.xlabel('Features')
plt.ylabel('Total Coefficient Value')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
for x, y in zip(features, total_coefficients):
    plt.text(x, y + 0, f'{y:.2f}', ha='center', va='bottom', fontsize=10, color='black')
plt.show()



# In[52]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                   header=None)
iris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris.dropna(how='all', inplace=True)

# Standardize the features
X = StandardScaler().fit_transform(iris.iloc[:, 0:4].values)

# Compute covariance matrix
covariance_matrix = np.cov(X.T)

# Compute eigenvalues and eigenvectors
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

# Sort eigenvalues and corresponding eigenvectors by descending order of eigenvalues
sorted_indices = np.argsort(eigen_values)[::-1]
eigen_values = eigen_values[sorted_indices]
eigen_vectors = eigen_vectors[:, sorted_indices]

# Project data onto the first two principal components
projection_matrix = eigen_vectors[:, :2]
X_pca = X.dot(projection_matrix)

# Extract principal component coefficients
PC1_coefficients = projection_matrix[:, 0]
PC2_coefficients = projection_matrix[:, 1]

# Plot biplot
plt.figure(figsize=(10, 8))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Plot feature loadings as arrows
for i in range(len(PC1_coefficients)):
    plt.arrow(0, 0, PC1_coefficients[i], PC2_coefficients[i], color='k', width=0.005, head_width=0.05)
    plt.text(PC1_coefficients[i] * 1.1, PC2_coefficients[i] * 1.1, iris.columns[i], color='k', ha='center', va='center')

# Plot samples colored by species
species_colors = {'Iris-setosa': 'g', 'Iris-versicolor': 'b', 'Iris-virginica': 'r'}
for species, color in species_colors.items():
    idx = iris['species'] == species
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], c=color, label=species, alpha=0.8, s=100)

plt.legend(title='Species')
plt.title('Biplot of PCA on Iris Dataset')
plt.grid(True)
plt.show()


# In[61]:


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Function to evaluate classifiers
def evaluate_classifier(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Function to compare models and select the best one
def compare_models(X_train, X_test, y_train, y_test):
    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    results = {}
    for name, clf in classifiers.items():
        accuracy = evaluate_classifier(clf, X_train, X_test, y_train, y_test)
        results[name] = accuracy

    # Find the best model based on accuracy
    best_model = max(results, key=results.get)
    return best_model, results

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Perform PCA (reduce to 2 components)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# Compare models before and after PCA
best_model_no_pca, results_no_pca = compare_models(X_train_std, X_test_std, y_train, y_test)
best_model_with_pca, results_with_pca = compare_models(X_train_pca, X_test_pca, y_train, y_test)

# Display results
print("Classification Results Without PCA:")
results_no_pca_df = pd.DataFrame(results_no_pca.items(), columns=['Classifier', 'Accuracy'])
print(results_no_pca_df)

print("\nClassification Results With PCA (2 Components):")
results_with_pca_df = pd.DataFrame(results_with_pca.items(), columns=['Classifier', 'Accuracy'])
print(results_with_pca_df)

# Plot results
plt.figure(figsize=(12, 6))

# Plot bar chart
plt.subplot(1, 2, 1)
plt.barh(results_no_pca_df['Classifier'], results_no_pca_df['Accuracy'], color='skyblue', label='No PCA')
plt.barh(results_with_pca_df['Classifier'], results_with_pca_df['Accuracy'], color='salmon', label='With PCA')
plt.xlabel('Accuracy')
plt.title('Classifier Performance Before and After PCA')
plt.legend()

# Plot table
plt.subplot(1, 2, 2)
plt.axis('off')  # Hide axes
plt.table(cellText=results_with_pca_df.values,
          colLabels=results_with_pca_df.columns,
          cellLoc='center',
          loc='center')

plt.tight_layout()
plt.show()

# Display the best model based on comparison
print(f"\nBest Model Before PCA: {best_model_no_pca} (Accuracy: {results_no_pca[best_model_no_pca]:.4f})")
print(f"Best Model After PCA: {best_model_with_pca} (Accuracy: {results_with_pca[best_model_with_pca]:.4f})")


# In[63]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target
target_names = data.target_names

# Standardize features
X = StandardScaler().fit_transform(X)

# Perform PCA
covariance_matrix = np.cov(X.T)
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
projection_matrix = eigen_vectors[:, :2]
X_pca = X.dot(projection_matrix)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Define Logistic Regression model
model = LogisticRegression()

# Define hyperparameters to tune
param_grid = {
    'penalty': ['l2'],  # Use only 'l2' penalty with lbfgs solver
    'C': [0.001, 0.01, 0.1, 1, 10, 100]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

# Perform grid search
grid_search.fit(X_train, y_train)

# Get the best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Evaluate the best model on test data
y_pred = best_model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Get classification report
class_report = classification_report(y_test, y_pred, target_names=target_names)

# Print results
print("Best Hyperparameters:", best_params)
print("Test Accuracy:", accuracy)
print("Classification Report:\n", class_report)


# In[64]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target
target_names = data.target_names

# Standardize features
X = StandardScaler().fit_transform(X)

# Perform PCA
covariance_matrix = np.cov(X.T)
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
projection_matrix = eigen_vectors[:, :2]
X_pca = X.dot(projection_matrix)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Define and train logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predict on test set
y_pred = logreg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=target_names)

# Print results
print("Test Accuracy:", accuracy)
print("Classification Report:\n", class_report)

# Plot decision boundaries
h = 0.02  # Step size in the mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Set1, alpha=0.8)
for species in range(len(target_names)):
    plt.scatter(X_pca[y == species, 0], X_pca[y == species, 1], label=target_names[species])

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Logistic Regression on PCA-transformed Iris dataset')
plt.legend()
plt.show()


# In[66]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target
target_names = data.target_names

# Standardize features
X = StandardScaler().fit_transform(X)

# Perform PCA
covariance_matrix = np.cov(X.T)
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
projection_matrix = eigen_vectors[:, :2]
X_pca = X.dot(projection_matrix)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Define and train KNN classifier
k = 3  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Predict on test set
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=target_names)

# Print results
print("Test Accuracy:", accuracy)
print("Classification Report:\n", class_report)

# Plot decision boundaries
h = 0.02  # Step size in the mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Set1, alpha=0.8)
for species in range(len(target_names)):
    plt.scatter(X_pca[y == species, 0], X_pca[y == species, 1], label=target_names[species])

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('KNN Classifier (k=3) on PCA-transformed Iris dataset')
plt.legend()
plt.show()


# In[67]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target
target_names = data.target_names

# Standardize features
X = StandardScaler().fit_transform(X)

# Perform PCA
covariance_matrix = np.cov(X.T)
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
projection_matrix = eigen_vectors[:, :2]
X_pca = X.dot(projection_matrix)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Define and train QDA classifier
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

# Predict on test set
y_pred = qda.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=target_names)

# Print results
print("Test Accuracy:", accuracy)
print("Classification Report:\n", class_report)

# Plot decision boundaries
h = 0.02  # Step size in the mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = qda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Set1, alpha=0.8)
for species in range(len(target_names)):
    plt.scatter(X_pca[y == species, 0], X_pca[y == species, 1], label=target_names[species])

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Quadratic Discriminant Analysis (QDA) on PCA-transformed Iris dataset')
plt.legend()
plt.show()


# In[68]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target
target_names = data.target_names

# Standardize features
X = StandardScaler().fit_transform(X)

# Perform PCA
covariance_matrix = np.cov(X.T)
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
projection_matrix = eigen_vectors[:, :2]
X_pca = X.dot(projection_matrix)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Define and train logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predict on test set
y_pred = logreg.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Logistic Regression (PCA-transformed Iris)')
plt.show()


# In[69]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target
target_names = data.target_names

# Standardize features
X = StandardScaler().fit_transform(X)

# Perform PCA
covariance_matrix = np.cov(X.T)
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
projection_matrix = eigen_vectors[:, :2]
X_pca = X.dot(projection_matrix)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Define and train KNN classifier
k = 3  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Predict on test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix for KNN (k={k}) on PCA-transformed Iris')
plt.show()


# In[70]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target
target_names = data.target_names

# Standardize features
X = StandardScaler().fit_transform(X)

# Perform PCA
covariance_matrix = np.cov(X.T)
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
projection_matrix = eigen_vectors[:, :2]
X_pca = X.dot(projection_matrix)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Define and train QDA classifier
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

# Predict on test set
y_pred = qda.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for QDA on PCA-transformed Iris')
plt.show()


# In[71]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target

# Standardize features
X = StandardScaler().fit_transform(X)

# Perform PCA
covariance_matrix = np.cov(X.T)
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
projection_matrix = eigen_vectors[:, :2]
X_pca = X.dot(projection_matrix)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Define and train logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Get predicted probabilities for each class
y_score = logreg.predict_proba(X_test)

# Binarize the labels for multi-class ROC curve
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_score[:, i])

# Plot ROC curve for each class
plt.figure(figsize=(8, 6))
colors = ['blue', 'green', 'red']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve (area = {roc_auc[i]:.2f}) for class {i}')

plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Logistic Regression (PCA-transformed Iris)')
plt.legend(loc='lower right')
plt.show()


# In[74]:


pip install shap


# In[ ]:




