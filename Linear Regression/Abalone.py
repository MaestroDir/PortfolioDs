"""
Abalone Dataset: Predict the Ring age in years MATH5836 Assignment 1

Written by Christopher Vaccaro (z5115339) on 24/09/2021
Some code assisted though:
 https://edstem.org/au/courses/6212/lessons/13871/slides/111793
 https://edstem.org/au/courses/6212/lessons/13871/slides/110007
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import linear_model, preprocessing
from sklearn.linear_model import LinearRegression
from sklearn import svm, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Get Dataset and clean such that there are no missing values or columns with non-numerical values
def get_data():
    # Data from UCI machine learning repository: https://archive.ics.uci.edu/ml/datasets/abalone
    # Data has already been cleaned from a set with missing values
    # However, "Sex" still has strings to identify what 
    
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
    col_names = ('Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
            'Viscera weight', 'Shell weight', 'Rings')
    # May want to find another way to do this other than manual entry, Very lengthy Process and could = erros
    data = pd.read_csv(url, header=None, names=col_names)


    # Calling data.describe() shows that in min value of 'Height' 0.00 which is likely incorrect data entry
    idx = data[data["Height"] == 0]
    data.drop(idx.index, inplace=True)
    
    # Fixing data["Sex"] column to have numerical values
    data['Sex'] = np.where(data['Sex'] == 'M', 0, np.where(data['Sex'] == 'F', 1, -1))
        
    return data

#Q2)
def corr_matrix(data):
    # creates correlation scores for variables
    corr_matrix = data.corr()
    
    # Not familiar with seaborn so code taken from: https://edstem.org/au/courses/6212/lessons/13871/slides/111828
    # annot = True to print the values inside the square
    sns.heatmap(data=corr_matrix, annot=True)
    plt.savefig('corr.png')

    plt.figure(figsize=(20, 5))

    # Q2) See that:
    # 1) Shell weight seems to have the highest positive correlation with our target variable rings
    # 2) Diameter is the second highest correlation with rings
    # 3) Sex is our least correlated feature with rings
    # 4) There is no negative correlation with rings so when selecting feature variables for our model we should use two 
    # highly correlated positive variable (if we are using 2 features to form our model)

# Q3) 
def scatter_plot(features, target, data):
    # Scatter plot code credited to https://edstem.org/au/courses/6212/lessons/13871/slides/111828
    for i, col in enumerate(features):
        plt.subplot(1, len(features) , i+1)
        x = data[col]
        y = target
        plt.scatter(x, y, marker='o')
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel('Rings')
        plt.savefig('feature.png')

    # Shell weight scatter plot shows that  for lower shell weights(g) we expect to see the number of rings (and hence age) of the 
    # abalone to be lower. For higher shell weights we should see more abalone with more rings

    # Similarly for our Diameter scatter plot we observe similar findings to shell weight. As Diameter of the abalone increases
    # we should expect in increase in the number of rings.

# Q4)
def hist_plot(name, data):
    plt.figure()
    plt.hist(data)
    plt.title(name)
    plt.savefig(name+'.png', dpi = 500)

# Q5)
def linear_model(normalize, data_x, data_y, t_size, experiment_no):
    
    if normalize != 1:
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=t_size, random_state=experiment_no)

        lin_model = LinearRegression()
        lin_model.fit(x_train, y_train)

        # model evaluation for training set (non-normalized data)
        y_test_predict = lin_model.predict(x_test)
        rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
        r2 = r2_score(y_test, y_test_predict)

        print("The model performance for test set (non-normalized data)")
        print("--------------------------------------")
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}'.format(r2))
        print("\n")
        
    else:
        # with normalisation
        X = preprocessing.scale(data_x)
        Y = preprocessing.scale(data_y)
        
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=t_size, random_state=experiment_no)
        
        lin_model = LinearRegression()
        lin_model.fit(x_train, y_train)
        
        # model evaluation for training set (normalized data)
        y_test_predict = lin_model.predict(x_test)
        rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
        r2 = r2_score(y_test, y_test_predict)

        print("The model performance for test set (normalized)")
        print("--------------------------------------")
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}'.format(r2))
        print("\n")   

#Q6)
def sex_diff(data):
    f, ax = plt.subplots(figsize=(10, 10))
    fig = sns.boxenplot(x='Sex', y="Rings", data=data)
    plt.show();

# Q7 Modeling
# Requires random selection of test sample and 30 experiments
# Visualisation of model
def scikit_linear_mod(normalize, data_x, data_y, t_size, experiment_no):
    
    if normalize != 1:
        
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=t_size, random_state=experiment_no)

        lin_model = LinearRegression()
        lin_model.fit(x_train, y_train)

        # model evaluation for training set (non-normalized data)
        y_test_predict = lin_model.predict(x_test)
        rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
        r2 = r2_score(y_test, y_test_predict)
        
        # Collect residuals
        residuals = y_test_predict - y_test

        print("The model performance for test set (non-normalized data)")
        print("--------------------------------------")
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}'.format(r2))
        print("\n")
        
        return residuals, rmse, r2
        
    else:
        # with normalisation
        X = preprocessing.scale(data_x)
        Y = preprocessing.scale(data_y)
        
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=t_size, random_state=experiment_no)
        
        lin_model = LinearRegression()
        lin_model.fit(x_train, y_train)
        
        # model evaluation for training set (non-normalized data)
        y_test_predict = lin_model.predict(x_test)
        rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
        r2 = r2_score(y_test, y_test_predict)
        
        # Collect residuals
        residuals = y_test_predict - y_test

        print("The model performance for test set (normalized)")
        print("--------------------------------------")
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}'.format(r2))
        print("\n") 
        
        return residuals, rmse, r2

def main():

    #Q1.1 
    data = get_data()
    data_x = data.drop("Rings", axis=1).values
    data_y = data["Rings"].values  

    
    #Q1.2 - note: normalizing data doesnt appear to significanly affect correlation
    corr_matrix(data)
    
    #Q1.3
    features = ['Diameter', 'Shell weight']
    target = data['Rings']
    scatter_plot(features, target, data)
    
    #Q1.4
    variables = ["Diameter", "Shell weight", "Rings"]
    for x in variables:
        hist_plot(x, data[x])
    
    #Q1.5 / Q2.4
    experiment_no = 30 # Change this if needed
    normalize = True # Change to TRUE for normalised data
    t_size = 0.4
    data_x_lm = data[['Diameter', 'Shell weight']].values      
    
    for i in range(experiment_no):
        print("Test No. {}".format(i+1))
        linear_model(normalize, data_x_lm, data_y, t_size, experiment_no)
    
    # Normalized
    # RMSE is 0.7740419250500538
    # R2 score is 0.4045876674100407
    
    # Non-normalized
    # RMSE is 2.4953432502120148
    # R2 score is 0.4045876674100407
    
    # Q2.1 / 2.2
    rmse_list = np.zeros(experiment_no)
    rsq_list = np.zeros(experiment_no)
    
    for i in range(experiment_no):
        
        print("Test No. {}".format(i+1))
        residuals, rmse, r2  = scikit_linear_mod(normalize, data_x, data_y, t_size, experiment_no)
        
        # Graph i: code example taken from Exercise 1.4 Part I Solution
        # https://edstem.org/au/courses/6212/lessons/13871/slides/111793
        plt.plot(residuals, linewidth=1);
        plt.savefig('scikit_linear{}.png'.format(i))
        
        # append
        rmse_list[i] = rmse
        rsq_list[i] = r2 
    
    for i in range(experiment_no):
        print("Test {} yields {} RMSE and {} R-squared score".format(i+1, rmse_list[i], rsq_list[i]))
        
    mean_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list)

    mean_rsq = np.mean(rsq_list)
    std_rsq = np.std(rsq_list)

    print(mean_rmse, ' mean_rmse',  std_rmse,  'std_rmse')
    print(mean_rsq, ' mean_rsq',  std_rsq,  'std_rsq')

    
    # Q6) Observing our boxenplot we can observe no significant difference between M F rings but the difference is due to
    # adult and infant
    # sex_diff(data)

if __name__ == '__main__':
    main()
