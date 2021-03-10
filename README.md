# pipelines-with-sklearn
datasets examples with sklearn pipelines
# Sklearn Pipeline:
This is a basic pipeline implementation. In real-life data science, scenario data would need to be prepared first then applied pipeline for rest processes. Building quick and efficient machine learning models is what pipelines are for. Pipelines are high in demand as it helps in coding better and extensible in implementing big data projects. Automating the applied machine learning workflow and saving time invested in redundant preprocessing work.

# Advantages of using Pipeline:
Automating the workflow being iterative.

Easier to fix bugs

Production Ready

Clean code writing standards

Helpful in iterative hyperparameter tuning and cross-validation evaluation

# challenges and solutions with sklearn pipelines
https://www.neuraxio.com/blogs/news/whats-wrong-with-scikit-learn-pipelines
https://www.neuraxle.org/stable/scikit-learn_problems_solutions.html#problem-defining-the-search-space-hyperparameter-distributions  

# All classification algs
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm.classes import OneClassSVM
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.neighbors.classification import RadiusNeighborsClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier    
from sklearn.gaussian_process.gpc import GaussianProcessClassifier
from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB  
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.mixture import DPGMM
from sklearn.mixture import GMM 
from sklearn.mixture import GaussianMixture
from sklearn.mixture import VBGMM


