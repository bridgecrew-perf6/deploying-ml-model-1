# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Hailu Teju Takele created the model. It is ada boost classifier using the 
default hyperparameters in scikit-learn 1.0.2.

## Intended Use
This model should be used to predict whether a person makes at least $50K a 
year salary or less based on a handful of features including age, level of 
education, marital status, class of work, and a few other attributes. The 
users of this model could be, for instance, non-profits organizations that 
servive on donations, and this model would allow them to target individual 
donors who are more likely to give to their organization.

## Training Data
80% of the cleaned version of the original data was used for training. 
Logarithmic transformation has been applied to a couple of numerical 
features that were highly skewed and MinMax scaler from scikit-learn was 
used to scale the remaining numerical features for better performance of the 
ML model used. Furthermore, Pandas' `get_dummies()` method was used to 
one-hot encode categorical features.

## Evaluation Data
20% of the cleaned version of the original data was used for testing and 
those same transformations that were done on the training set have been 
applied to the testing set as well.

## Metrics
`f_beta` score with `beta = 0.5` was used to score the trained model's 
predictions on both the training set and testing set. The scores were: `0.
7354` for the training set and `0.7127` for the testing set.

## Ethical Considerations
The dataset used here is base a US census data that is openly available. 
Still, care should be taken in what type of information to be shared with 
prospective donors while reaching out for their donations. For instance, an 
explicity mention of a feature learned used by the model during model 
training would be a nonstarter for ethical reasons.

## Caveats and Recommendations
Some features in the dataset are highly skewed and have values that tend to lie 
near a single number, but also a significant number of values that are very 
large or very small compared to the high frequency value. The `capital-gain` 
and `capital-loss` attributes in the dataset are such features. Algorithms can 
be sensitive to such distributions of values and can underperform if the range 
is not properly normalized. That's why we performed log-transformations on 
these features. Also note that when applying a log-transformation, one 
should keep in mind that the logarithmic function is undefined at zero. 
Hence, small perturbation of values may be required before applying a 
logarithmic transformation. 

Even though, we chose ada boost classifier for our model, there are many 
other algorithms that could be used for classification. Also, we have not 
done any hyper-parameter tuning in this project and that can be used to 
further optimize hyper-parameters and obtain better prediction scores.
