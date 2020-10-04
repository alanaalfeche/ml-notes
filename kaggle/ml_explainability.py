'''Permutation Importance

Feature importance answers the question which features have the biggest impact on prediction? 

One measurement is through permutation importance. It is calculated after a model has been fitted. 
The main idea is to shuffle the values in a single column and measure the loss function suffered from shuffling. 

'''
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, DecisionTreeClassifier

X = 'somefeatures'
y = 'sometarget'
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=100,random_state=0).fit(train_X, train_y)
perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
# returns weight and feature where values towards the top are the most important features 
eli5.show_weights(perm, feature_names = val_X.columns.tolist())

'''Partial Plots

Partial Dependence Plots show how a feature affects predictions. Calculated after model has been fitted. 
The idea is to repeateadly alter the value for one variable to make a series of predictions. 
'''
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots
from sklearn import tree
import graphviz

my_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)
# tree visualization of the features 
tree_graph = tree.export_graphviz(my_model, out_file=None, feature_names='feature_names')
graphviz.Source(tree_graph)
# create pdp data
data_to_plot = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features='feature_names', feature='feature_to_plot')
pdp.pdp_plot(data_to_plot, 'title')
# y-axis represents the change in prediction, shaded area indicates level of confidence
plt.show()

# 2D PDP -- visualizes the interactions between features
features_to_plot = ['col1', 'col2']
inter1  =  pdp.pdp_interact(model=my_model, dataset=val_X, model_features='feature_names', features=features_to_plot)
pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')
plt.show()

'''SHAP Values

Short for SHapley Additive exPlanations, SHAP can explain individual prediction by the impact of each feature
Advanced uses of SHAP Value here: https://www.kaggle.com/dansbecker/advanced-uses-of-shap-values
'''
import shap

data_for_prediction = val_X.iloc[5]
explainer = shap.TreeExplainer(my_model) # shap.DeepExplainer = deep learnign models or shap.KernelExplainer = for all models
# shap_values object above is a list with two arrays. The first array is the SHAP values for a negative outcome and the second array is the list of SHAP values for the positive outcome
shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
