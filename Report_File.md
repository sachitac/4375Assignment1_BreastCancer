# Report File

I am satisfied with my solution because it yielded an r^2 value of 0.75 which is indicative of a strong relationship between the independent variables (the features) and the dependent variables
(the targets). The r^2 value also captures a significant variability which implies the model has a good predictive ability and fits the data well. Furthermore the mse value 
produced was 0.057 which is a good value for a binary classification.  But because this is a binary classification the linear regression is not the best way to calculate the regression, 
but given our linear model's results, I am satisfied with our program.

I have included a picture of both the graphs we used to visualize the program. 

The heatmap below represents the correlation between different features in the breast cancer dataset, with values ranging from -1 (strong negative correlation) 
to 1 (strong positive correlation). Warmer colors indicate positive correlations, while cooler colors indicate negative or no correlation between the features.
The conclusion of the heatmap is that certain features in the breast cancer dataset are highly correlated with each other, either positively or negatively. 
![heatmap](https://github.com/sachitac/4375Assignment1_BreastCancer/blob/main/heatmap.png)

The feature and value graph below represents the relationship between specific features (variables) and their corresponding values in the dataset. 
It visualizes how different feature values are distributed, possibly helping to identify patterns or trends in the data that could be useful for predicting tumor diagnosis. 
The conclusion of the graph is that it shows how individual feature values are related to the target variable (tumor diagnosis). 
By examining these relationships, I can gain insights into which features have stronger associations with the outcome, which can inform feature selection and model refinement.
![feature and value graph](https://github.com/sachitac/4375Assignment1_BreastCancer/blob/main/feature%20and%20value%20graph.png)

The logs to our trials is included in the model_log.txt file located in this repository. 





