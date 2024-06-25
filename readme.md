![alt text](image.png)

<h1> Kaggle XSkill Assessment Challenge</h1> 
The goal of this challenge is to predict the prices of cars using the features given in the synthetic dataset inspired by the original car prices dataset on Kaggle.The target is to obtain the Lowest MSE (Mean Squared Error) possible.
<br></br>
It is represented as:

RMSE = $ \sqrt{{(\frac{1}{N} \sum_{i=1}^{N} (y_i - \widehat{y}_i)^2)}}$
<br></br>
where $\hat{y}_i$ is the predicted value and $y_i$ is the original value for each instance i.
<br></br>

***My submission was able to get to a rank of 94 out of more than 14.5K submissions.*** 





<h2>Files</h2>

* train.csv - the training dataset
* test.csv - the test dataset
* cleaned_train_data.csv - training data cleaned and processed
* predicted.csv - all the predicted prices for the given features in test data
* data_clean_pipeline.py - has the data cleaning pipeline function 
* trainingmodel.py - contains code on how to train the model


<h2>Model Used</h2>

After comparing a few base regression  models , the initial MSE was the lowest for *HistGradientRegressor* model . So I finetuned the hyperparameters using *RandomisedCV* search algorithm .
