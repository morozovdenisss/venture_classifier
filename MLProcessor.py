import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn import preprocessing
from sklearn import svm, metrics
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression as lm
from sklearn.linear_model import RidgeCV
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_validate, RepeatedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import csv
from pandas import DataFrame, ExcelWriter
import os
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

"""
Knowledge base
https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html - Good Article

"""

# Create automatic Min Max Scaling between 0 - 1 in the future. Excel is much faster.


# Open the file and separate into table by ";"
data=pd.read_csv('CSV_ML_Exits_minmax.csv', sep=';', dtype=None)
data=data.dropna()
#print (data.head)

# Split into labels and features - drop Successful as a target in Y
y = data.Successful
#X = data.drop(data.columns[[1,72]], axis = 1)
X = data.iloc[: , :-1]
X_Headers = list(data.columns.values)
X_Headers = X_Headers[:-1]

# Open Pandas file and keep writing items
ImpurityPermutation_Bar  = 'ExportImpurityPermutation_Bar.csv'
FeatureImportance__Bar = 'ExportFeatureImportance_Bar.csv'
# For Pairplots
All_Headers = list(data.columns.values)
'''
Redacted
'''

# Split by 20% train/test
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, stratify=y)
 
#Run prints below if the code breaks
'''
print(X_train.shape)
print(X_test.shape)
print('Xheaders', X_Headers)
print('X', X.shape)
'''

# Simple Linear Regression
def SLR():
    model=lm().fit(X_train,y_train)
    predictions=model.predict(X_test)
    plt.scatter(y_test,predictions)
    plt.xlabel('True values')
    plt.show() 

#Confusion matrix - testing with Error Tolerance https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def Confusion_FromEstimator():
    try:
        classifier = svm.SVC(kernel="linear", C=0.1).fit(X_train, y_train)
        np.set_printoptions(precision=2)
        titles_options = [
            ("Confusion matrix, without normalization", None),
            ("Normalized confusion matrix", "true"),
        ]
        for title, normalize in titles_options:
            disp = ConfusionMatrixDisplay.from_estimator(
                classifier,
                X_test,
                y_test,
                display_labels=y,
                cmap=plt.cm.Blues,
                normalize=normalize,
            )
            disp.ax_.set_title(title)
        plt.show()
    except:
        pass

def Confusion_FromPrediction():
    clf = SVC(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(
       y_test, y_pred)
    plt.show()


#Feature Importance
# Can be misleading for high cardinality features
def ImpurityPermutation():
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    
    #Plot
    forest_importances = pd.Series(importances, index=X_Headers)   
    forest_importances = forest_importances.sort_values(ascending=False)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Impurity Permutation using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.set_dpi(100)
    plt.plot(std)

    # Scoring & Exporting Datast)
    score = forest.score(X_test, y_test)
    print ('ImpurityPermutation Acuracy: ', score)
    df = pd.DataFrame(forest_importances)
    df.to_csv(ImpurityPermutation_Bar)

# The permutation based importance can be used to overcome drawbacks of default feature importance computed with mean impurity decrease
# The permutation based importance is computationally expensive. The permutation based method can have problem with highly-correlated features, it can report them as unimportant.

def FeatureImportance_Boxplot():
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_train, y_train)
    
    result = permutation_importance(
        forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    sorted_idx = result.importances_mean.argsort()
    
    fig, ax = plt.subplots()
    ax.boxplot(
        result.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx]
    )
    ax.set_title("Feature Importance - Boxplot")
    fig.set_size_inches(11.5, 17.5, forward=True)
    fig.set_dpi(200)
    plt.show()
    
    # Scoring & Exporting Datast)
    score = forest.score(X_test, y_test)
    print ('Feature Importance - Boxplot Accuracy: ', score)

def FeatureImportance_Bar():
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_train, y_train)

    result = permutation_importance(
        forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )     
    forest_importances = pd.Series(result.importances_mean, index=X_Headers)
    forest_importances = forest_importances.sort_values(ascending=False)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature Importance - Bar Chart")
    ax.set_ylabel("Mean accuracy decrease")
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.set_dpi(100)
        
    #np.savetxt("1234.csv", np.c_[x,y])
    score = forest.score(X_test, y_test)
    print ('Feature Importance - Bar Accuracy: ', score)
    #print ('Feature Importance - Bar Chart: ', forest_importances)
    
    df = pd.DataFrame(forest_importances)
    df.to_csv(FeatureImportance__Bar)

# Correlation Scatterplot + Histograms (without independent variable)
def correlation():
    # FIGURE OUT HUE KEY ERROR with SNS
    map = sns.pairplot(sns_data, plot_kws={'alpha':0.5, 'edgecolor': 'k'})
    map.map_diag(plt.hist)
    map.map_upper(sns.kdeplot, shade =True)

# Variability of the Coefficients
# We can check the coefficient variability through cross-validation: it is a form of data perturbation (related to resampling).
# If coefficients vary significantly when changing the input dataset their robustness is not guaranteed, and they should probably be interpreted with caution.
# Increase in certain feature means that much increase in importance if positive, and decrease if negative

def RidgeCoef():
    model = RidgeCV()
    model.fit(X_train, y_train)
    coefs = pd.DataFrame(
       model.coef_,
       columns=['Coefficients'], index=X_Headers
    )
    #print (coefs)
    coefs = coefs.sort_values(by='Coefficients')
    df = pd.DataFrame(coefs)
    df.to_csv('ExportRidgeCoef.csv')
    
    coefs.plot(kind="barh", figsize=(26, 24))
    plt.title("Ridge model, small regularization")
    plt.axvline(x=0, color=".5")
    plt.subplots_adjust(left=0.2)


def coefvariability():
    model = make_pipeline(StandardScaler(), RidgeCV())
    model.fit(X_train, y_train)
    cv_model = cross_validate(
       model, X_test, y_test, cv=RepeatedKFold(n_splits=5, n_repeats=5),
       return_estimator=True, n_jobs=2
    )
    coefs = pd.DataFrame(
       [model[1].coef_
        for model in cv_model['estimator']],
       columns=X.columns
    )
    meds = coefs.median()
    meds.sort_values(ascending = False, inplace = True)
    coefs = coefs[meds.index]
    #print (meds.index)
    plt.figure(figsize=(20, 18))
    sns.boxplot(data=coefs, orient='h', color='cyan', saturation=0.5)
    plt.axvline(x=0, color='.5')
    plt.xlabel('Coefficient importance')
    plt.title('Coefficient importance and its variability')
    plt.subplots_adjust(left=.3)
    #print('Coefvariability Array of 3: ', cross_val_score(model, X, y, cv=3))
    score = model.score(X_test, y_test)
    #print('Coef', coefs)
    print ('Coefvariability Score cross_val: ', score)
    
    #Validation Performance
    result = permutation_importance(model, X_test, y_test, n_repeats=10,
                                    random_state=0)
    
    # Export Data in a Hacky Way because dimensions are different
    a = result.importances_mean
    b = result.importances_std
    c = result.importances
    
    np.savetxt('ExportCoef-Mean.csv', a)
    np.savetxt('ExportCoef-STD.csv', b)
    np.savetxt('ExportCoef-Importances.csv', c)
    
    writer = ExcelWriter("ExportCoefVariability_Combined.xlsx")
    names = ['ExportCoef-Mean.csv', 'ExportCoef-STD.csv', 'ExportCoef-Importances.csv']
    for filename in names:
        df_csv = pd.read_csv(filename)
    
        (_, f_name) = os.path.split(filename)
        (f_shortname, _) = os.path.splitext(f_name)
    
        df_csv.to_excel(writer, f_shortname, index=False)
    writer.save()


def launch():
    #Confusion Matrix
    Confusion_FromEstimator()
    Confusion_FromPrediction()
    
    #Feature Importance Calculations
    ImpurityPermutation()
    FeatureImportance_Boxplot()
    FeatureImportance_Bar()
    RidgeCoef()
    coefvariability()
    correlation()

#for i in range(1,10):
launch()


'''
# Comparison Classifier- used it to get understanding on KNeighbors
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
def ComparClass():
    h = 0.02  # clarify this one
    
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]
    
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]
    
    
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)
    
    datasets = [
        make_moons(noise=0.3, random_state=0),
        make_circles(noise=0.2, factor=0.5, random_state=1),
        linearly_separable,
    ]
    
    figure = plt.figure(figsize=(27, 9))
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
    
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
        # Plot the testing points
        ax.scatter(
            X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
        )
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1
    
        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
    
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
    
            # Plot the training points
            ax.scatter(
                X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
            )
            # Plot the testing points
            ax.scatter(
                X_test[:, 0],
                X_test[:, 1],
                c=y_test,
                cmap=cm_bright,
                edgecolors="k",
                alpha=0.6,
            )
    
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(
                xx.max() - 0.3,
                yy.min() + 0.3,
                ("%.2f" % score).lstrip("0"),
                size=15,
                horizontalalignment="right",
            )
            i += 1
    
    plt.tight_layout()
    plt.show()

    rf = RandomForestClassifier(random_state=0).fit(X, y)
    result = permutation_importance(rf, X, y, n_repeats=10, random_state=0, n_jobs=2)
    
    fig, ax = plt.subplots()
    sorted_idx = result.importances_mean.argsort()
    ax.boxplot(
        result.importances[sorted_idx].T, vert=False, labels=X_Headers[sorted_idx]
    )
    ax.set_title("Permutation Importance of each feature")
    ax.set_ylabel("Features")
    fig.tight_layout()
    plt.show()
    
# Dimensionality Reduction
# https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_dim_reduction.html#sphx-glr-auto-examples-neighbors-plot-nca-dim-reduction-py

def DimenRed():
    n_neighbors = 2
    random_state = 0
    #dim = len(X[0])
    #n_classes = len(np.unique(y))
    
    # Reduce dimension to 2 with PCA
    pca = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=random_state))
    
    # Reduce dimension to 2 with LinearDiscriminantAnalysis
    lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=2))
    
    # Reduce dimension to 2 with NeighborhoodComponentAnalysis
    nca = make_pipeline(
        StandardScaler(),
        NeighborhoodComponentsAnalysis(n_components=2, random_state=random_state),
    )
    
    # Use a nearest neighbor classifier to evaluate the methods
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Make a list of the methods to be compared
    dim_reduction_methods = [("PCA", pca), ("LDA", lda), ("NCA", nca)]
    
    # plt.figure()
    for i, (name, model) in enumerate(dim_reduction_methods):
        plt.figure(figsize=(8, 12), dpi=80)
        # plt.subplot(1, 3, i + 1, aspect=1)
        # Fit the method's model
        model.fit(X_train, y_train)
        # Fit a nearest neighbor classifier on the embedded training set
        knn.fit(model.transform(X_train), y_train)
        # Compute the nearest neighbor accuracy on the embedded test set
        acc_knn = knn.score(model.transform(X_test), y_test)
        # Embed the data set in 2 dimensions using the fitted model
        X_embedded = model.transform(X)
        # Plot the projected points and show the evaluation score
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap="Set1")
        plt.title(
            "{}, KNN (k={})\nTest accuracy = {:.2f}".format(name, n_neighbors, acc_knn)
        )
    plt.show()

'''
