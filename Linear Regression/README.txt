  Title:
Data processing and linear regression for MATH5836

  Description:
Students were given 24hrs to complete Assignment involving data processing and modelling of Abalone Dataset. They were then to write a report of their findings.
The age of Abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope. As this process is time consuming
students were to asked to see if other physical features offered predictive insights into the Abalones age.


    Data processing (1.5 Marks):
  1. Clean the data (eg. convert M and F to 0 and 1). You can do this with code or simple find and replace. 
  2. Develop a correlation map using a heatmap and discuss major observations.
  3. Pick two of the most correlated features (negative and positive) and create a scatter plot with ring-age. Discuss major observations. 
  4. Create histograms of the two most correlated features, and the ring-age. What are the major observations? 
  5. Create a 60/40 train/test split - which takes a random seed based on the experiment number to create a new dataset for every experiment. 
  6. Add any other visualisation of the dataset you find appropriate (OPTIONAL). 

    Modelling  (2.0 Marks):
  1. Develop a linear regression model using all features for ring-age using 60 percent of data picked randomly for training and remaining for testing. Visualise your model prediction using appropriate plots. Report the RMSE and R-squared score. 
  2. Develop a linear regression model with all input features, i) without normalising input data, ii) with normalising input data. 
  3. Develop a linear regression model with two selected input features from the data processing step. 
  4. In each of the above investigations, run 30 experiments each and report the mean and std of the RMSE and R-squared score of the train and test datasets. Write a paragraph to compare your results of the different approaches taken. Note that if your code can't work for 30 experiments, only 1 experiment run is fine. You won't be penalised if you just do 1 experiment run. 
  5.Upload your code in Python/R or both. The code should use relevant functions (or methods in case you use OOP) and then create required outputs. 

    Report  (1.5 Marks):
  1. Create a report and include the visualisations and results obtained and discuss the major trends you see in the visualisation and modelling by linear regression  
  2. Upload a pdf of the report. 

  Dataset:
https://archive.ics.uci.edu/ml/datasets/abalone
