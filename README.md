# Savor or Skip
## The Art of Recipe Rating Prediction
Using Machine Learning to find out what's worth eating


## Problem Identification
In this project, we will continue to build off the progress we made analyzing the recipes and reviews dataset from before. This time, we will explore building a predictive model to classify recipes. The classification task we are going to attempt is determining which recipes are *amazing!*  

We could try predicting the rating in terms of the number of stars, but since most of the reviews are 5 stars, we could easily achieve an accuracy of around 70% just by picking 5 every time. We have to choose some condition that recipes must satisfy to be considered *amazing.*

**1. Recipes with only 5-star reviews**: This condition will include recipes that are generally well-received, but could potentially include recipes that have been reviewed only once (thus, there might not be a consensus on their quality).

**2. Recipes with only 5-star reviews and more than one review**: This condition is more stringent and could exclude some good recipes that received a single non-perfect rating.

**3. Recipes that have been reviewed more than once, where every review gave a rating of at least 4 stars, and at least one review awarded the maximum 5 stars**: This seems like a good balance. It ensures that the recipe is consistently highly rated but doesn't require perfection in all reviews.

**4. Only recipes that have more than one review and then apply the condition of recipes with only 5-star reviews**: This condition attempts to find a balance by excluding recipes with only one review (which might not be representative) and then looking for consistently perfect scores. These will be the same recipes as in condition 2 but with a smaller overall dataset.

The condition we choose depends on what we want our model to be able to classify. We can think of lower proportions corresponding to 'higher standards' and the classifier will learn to identify 'better' recipes. For this project, the classifier should be trying to find the very best recipes, so we will go with condition 2. This will help us find out what sets those top recipes apart. Other conditions can be tried later to see how it affects models.  

We can create the target column by running the following code on the merged dataset:
```python
aggfuncs = {
    'minutes':'first',
    'n_steps':'first',
    'n_ingredients':'first',
    'ingredients':'first',
    'nutrition':'first',
    'rating':['min','count','mean']
    }

df = df.groupby('recipe_id').agg(aggfuncs)
df.columns = list(aggfuncs.keys()) + ['count','mean']

df = df[df['count'] > 1]
df['is_amazing'] = (df['rating'] == 5).astype(int)
```
* * *
<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th>recipe_id</th>\n      <th>minutes</th>\n      <th>n_steps</th>\n      <th>n_ingredients</th>\n      <th>ingredients</th>\n      <th>nutrition</th>\n      <th>rating</th>\n      <th>count</th>\n      <th>mean</th>\n      <th>is_amazing</th>\n      </thead>\n  <tbody>\n    <tr>\n      <th>275022.0</th>\n      <td>50</td>\n      <td>11</td>\n      <td>7</td>\n      <td>[cheddar cheese, macaroni, milk, eggs, bisquick, salt, red pepper sauce]</td>\n      <td>[386.1, 34.0, 7.0, 24.0, 41.0, 62.0, 8.0]</td>\n      <td>1.0</td>\n      <td>3</td>\n      <td>3.0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>275026.0</th>\n      <td>45</td>\n      <td>7</td>\n      <td>9</td>\n      <td>[frozen crabmeat, sharp cheddar cheese, cream cheese, onion, milk, bisquick, eggs, salt, nutmeg]</td>\n      <td>[326.6, 30.0, 12.0, 27.0, 37.0, 51.0, 5.0]</td>\n      <td>1.0</td>\n      <td>2</td>\n      <td>3.0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>275030.0</th>\n      <td>45</td>\n      <td>11</td>\n      <td>9</td>\n      <td>[apple pie filling, graham cracker crust, cream cheese, sugar, vanilla, eggs, caramel topping, pecan halves, pecans]</td>\n      <td>[577.7, 53.0, 149.0, 19.0, 14.0, 67.0, 21.0]</td>\n      <td>5.0</td>\n      <td>10</td>\n      <td>5.0</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>275035.0</th>\n      <td>5</td>\n      <td>5</td>\n      <td>7</td>\n      <td>[honey, peanut butter, powdered milk, chocolate chips, raisins, coconut, nuts]</td>\n      <td>[1908.7, 152.0, 801.0, 44.0, 133.0, 174.0, 71.0]</td>\n      <td>4.0</td>\n      <td>2</td>\n      <td>4.0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>275036.0</th>\n      <td>15</td>\n      <td>6</td>\n      <td>8</td>\n      <td>[frozen corn, green pepper, onion, garlic cloves, butter, olive oil, salt &amp; pepper, parmesan cheese]</td>\n      <td>[270.4, 20.0, 4.0, 3.0, 11.0, 37.0, 12.0]</td>\n      <td>5.0</td>\n      <td>2</td>\n      <td>5.0</td>\n      <td>1</td>\n    </tr>\n  </tbody>\n</table>

```python
df['is_amazing'].mean()
```
```
0.48655620597281474
```
About 50% of the data belongs to the *amazing* class which is going to be better for our model as the classes are much more balanced compared to the original ratings.  


We will attempt to predict if a recipe belongs to the *amazing* class based on several features:

- `minutes`: The total time it takes to prepare the recipe.
- `n_steps`: The number of steps involved in making the recipe.
- `n_ingredients`: The number of ingredients required by the recipe.
- `ingredients`: A list of the recipe's ingredients.
- `nutrition`: Nutritional information about the recipe, including calories, fat, sugar, sodium, protein, saturated fat, and carbohydrates.

Predicting based on these features will force our model to (hopefully) learn about what intrinsically makes a recipe good rather than using the user review text. We will quantify how good our model is with F1 score. F1 is a combination of recall and precision; recall is important especially for the positive class since we want the model to be able to identify a high proportion of the *amazing* recipes, and precision is also important since we don't want the model to achieve high recall by simply guessing positive for everything.  

F1 is a good balance of these two metrics and also better than accuracy, especially if we decide to use a different condition for *amazingness* that would make the classes less evenly distributed, similar to how they are in the original dataset. For example, if we vary the condition just slightly to include recipes with only 5-star reviews, but drop the condition that all recipes must have more than a single review, the positive class would only make up 24% of our data. This could lead to a relatively high accuracy but the model won't actually be good at predicting what we want.

## Baseline Model

The baseline model will be a simple model that doesn't use all of the features we selected to give us a ... you guessed it ... baseline of performance to compare future models against. To keep things simple, we won't worry about the nutrition column until later.  

The numerical features, `minutes`, `n_steps`, and `n_ingredients` will be passed through the pipeline un-transformed. To use ingredients as a feature, we can use the lists they are stored in as a pre-tokenized format for sklearn's `TfidfVectorizer` class. Here's what our pipeline looks like:  

```python
TfidfBaseline = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False, min_df=100)

baseline_preprocessor = ColumnTransformer(transformers=[
    ('', 'passthrough', ['minutes','n_steps','n_ingredients']),
    ('tfidf', TfidfBaseline, 'ingredients')
], verbose=True)


# Now we can create a pipeline that combines our preprocessor with our classifier
baseline_model = Pipeline(steps=[
    ('preprocessor', baseline_preprocessor),
    ('classifier', MultinomialNB()),
], verbose=True)
```

The `TfidfVectorizer` will take the tokenized corpus and return an array of the TF-IDF scores. The array will look very similar to a bag-of-words transformation, but each term carries relative importance to the overall corpus. The `min_df` parameter controls which words we eliminate from the vocabulary. Words that appear less time, aka ingredients in less recipes than the value of `min_df` will be excluded. This helps reduce the size of the vocabulary significantly from around 11,000 ingredients to several hundred to the low thousands depending on the value. Huge sparse matrices can consume lots of memory, imagine if we used all 11,000 ingredients and 200,000+ rows.  

#### About Naive Bayes  
For this model we are using sklearn's built in `MultinomialNB()` classifier. The core principle behind Naive Bayes is Bayes' Theorem, which calculates the probability of a certain event given prior knowledge. In text classification, we're interested in the probability of a document belonging to a certain class given the words in it, or in our case, the probability a recipe is *amazing* given the ingredients in it. The 'naive' part of Naive Bayes refers to the assumption that all features are independent of each other, which means the presence or absence of a word doesn't affect the presence or absence of any other word. This is a significant simplification that allows the model to scale well with the number of features and data points.  

Given a training set, the model calculates the prior probability of each class (i.e., the probability of each class in the training set), as well as the conditional probability of each word given each class (i.e., the probability of a word appearing in documents of a certain class).  

After a brief wait for the model to train, we can see its performance:  
```
Classification Report:
              precision    recall  f1-score   support

           0       0.55      0.27      0.37      4117
           1       0.50      0.77      0.61      3947

    accuracy                           0.52      8064
   macro avg       0.53      0.52      0.49      8064
weighted avg       0.53      0.52      0.49      8064


ROC AUC:
0.5518896879893714
```

The F1 scores aren't great, and our AUC is close to 0.5 meaning the model is having a tough time distinguishing between classes, but we can see the recall for class 1 (*amazing* class) is 0.77 which is actually quite good. This means the model is able to pick out a majority of the *amazing* recipes. Can we do better though? Let's move on.


## Final Model
For the final model, we are going to create some additional features out of the nutrition column to help us with predictions. The ingredients are stored as a list in the column, and the values in the list represent: `['cals', 'fat', 'sugar', 'sodium', 'protein', 'sat_fat', 'carbs']`.  

The goal of adding these features is to give the model more detailed information about the food itself. The more information that describes the food is available, the more we can learn from that information about what makes recipes *amazing*. However, this will only be the case if our model improves as a result of these new features. If we extensively test a multitude of models with the new features and can't find any noticeable improvement, we can probably conclude those features are unrelated to the target variable. In the case of nutrition levels, one might guess recipes with high levels of sugar would be rated higher (who doesn't love dessert :P).

We will define a custom class that inherits from sklearn's `BaseEstimator` and `TransformerMixin` so we can include this step in our final pipeline. Here's what the class and the pipeline look like:  
```python
class NutritionExpander(BaseEstimator, TransformerMixin):
    def __init__(self, quantile=0.99):
        self.nuts = ['cals', 'fat', 'sugar', 'sodium', 'protein', 'sat_fat', 'carbs']
        self.quantile = quantile

    def fit(self, X, y=None):
        self.limits = {nut: X.apply(lambda x: x[i]).quantile(0.99) for i, nut in enumerate(nuts)}
        return self

    def transform(self, X):
        X_expanded = pd.DataFrame(X.tolist())
        X_expanded.columns = nuts
        for col in X_expanded.columns:
            X_expanded[col] = X_expanded[col].clip(upper=self.limits[col])
        return X_expanded

    def get_feature_names_out(self, input_features=None):
        return self.nuts

# Create the Tfidf vectorizer
TfidfIngredients = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False, min_df=50)

# Create the column transformation pipeline
preprocessor = ColumnTransformer(transformers=[
    ('', 'passthrough', ['minutes','n_steps','n_ingredients']),
    ('nutrition', NutritionExpander(), 'nutrition'),
    ('tfidf', TfidfIngredients, 'ingredients')
],)

# Now we can create a pipeline that combines our preprocessor with our classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression()),
],)
```

We have used `LogisiticRegression()` as a placeholder model as we are going to do a grid search to try to find a better model and optimize its hyperparameters. We can use `GridSearchCV()` to test various combinations of estimators and their parameters. We can also choose the scoring metric for the grid search to optimize over, in our case it will be F1-weighted. We will set `cv=5` to perform 5 cross-validating folds for each combination. Here is the code to perform the search:  
```python
# Define models we will test
models = {
    'LogisticRegression': LogisticRegression(max_iter=10000),
    'GradientBoosting': GradientBoostingClassifier(),
    'LinearSVC': LinearSVC(),
    'MultinomialNB': MultinomialNB(),
    'AdaBoostClassifier': AdaBoostClassifier(),
}

# Define hyperparameters
param_grid = {
    'LogisticRegression': {
        'classifier__C': list(range(1, 10, 2)),
    },
    'GradientBoosting': {
        'classifier__n_estimators': list(range(10, 100, 20)),
        'classifier__learning_rate': [0.01, 0.1, 1],
    },
    'LinearSVC': {
        'classifier__C': list(range(1, 10, 2)),
    },
    'MultinomialNB': {
        'classifier__alpha': [0.01, 0.1, 1],
    },
    'AdaBoostClassifier': {
        'classifier__n_estimators': list(range(10, 100, 20)),
        'classifier__learning_rate': [0.01, 0.1, 1],
    },
}

model_dict = {}

# Run GridSearchCV for each model
for model_name in models:
    model.set_params(classifier=models[model_name])
    grid_search = GridSearchCV(model, {**param_grid[model_name]}, cv=5, scoring='f1_weighted', verbose=10, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for {model_name}: ", grid_search.best_params_)
    model_dict[model_name] = grid_search
```
Running this could take a while depending on the number of hyperparameters we choose to test, it can easily run away from you. A better strategy to try next time might be to run a randomized search first to get a general idea of what works well for this problem, then find the true optimal parameters with a grid search of a more clearly defined search space. The results from `model_dict` are:  
```python
LogisticRegression {'classifier__C': 9} 0.5498391251939745
GradientBoosting {'classifier__learning_rate': 1, 'classifier__n_estimators': 10} 0.5448388768551441
LinearSVC {'classifier__C': 1} 0.438448919853249
MultinomialNB {'classifier__alpha': 0.01} 0.5251873963510547
AdaBoostClassifier {'classifier__learning_rate': 1, 'classifier__n_estimators': 30} 0.5455777578280969
```
Our placeholder classifier turned out to be the best-performing model! Retraining on the entire training set with those settings for `LogisticRegression()` and evaluating on the test set gives us the following:  

```
[Pipeline] ...... (step 1 of 2) Processing preprocessor, total=   0.4s
[Pipeline] ........ (step 2 of 2) Processing classifier, total=  11.6s

Accuracy:
0.5425347222222222

Confusion Matrix:
[[2458 1659]
 [2030 1917]]

Classification Report:
              precision    recall  f1-score   support

           0       0.55      0.60      0.57      4117
           1       0.54      0.49      0.51      3947

    accuracy                           0.54      8064
   macro avg       0.54      0.54      0.54      8064
weighted avg       0.54      0.54      0.54      8064
```
Our F1-score did increase slightly, which we hope is the case as the grid search was optimizing for it. We can say overall, the model got slightly better, but if we look at the class 1 recall score, it actually dropped quite significantly meaning the model is not able to identify as many of the *amazing* recipes as the baseline. I would like to try and figure out if it's possible to optimize for a metric for a specific class in classification problems with the tools provided by sklearn in the future. Perhaps a better choice of the scoring metric would lead to different results, but the search did what we asked and found a model that improved F1.  

#### About Logistic Regression  
Logistic regression is a statistical model used for binary classification problems, which are problems with two possible outcomes. Despite its name, logistic regression is an algorithm for classification, not regression.

The logistic regression model computes a weighted sum of the input features (plus a bias term), but instead of outputting the result directly like a linear regression model, it outputs the logistic of this result. The logistic is a sigmoid function that outputs a number between 0 and 1. It is used to convert the output of the linear part of the model into a probability, which can be used to make a binary prediction.

Here's a step-by-step description of how it works:

**1. Combining the inputs with weights**: Each feature in the dataset is assigned a weight (or coefficient), which can be interpreted as the importance of the feature. A linear equation is then formed which is a weighted sum of the features. It also includes an extra bias term (also known as the intercept term).

**2. Applying the logistic function**: The result of the linear equation is then fed to the logistic function (also known as the sigmoid function). The sigmoid function is an S-shaped curve that can take any real-valued number and map it into a value between 0 and 1.

**3. Making a prediction**: A threshold is chosen, usually 0.5. If the output probability is above the threshold, the model predicts class 1, otherwise, it predicts class 0.

**4. Estimating the parameters**: The parameters (weights and bias) of the model are estimated using a training dataset. This is done by maximizing the likelihood of the observed data, which leads to the minimization of the cost function (also known as log loss). This process is often carried out using an optimization algorithm like Gradient Descent.

The result is a model that can estimate the probability of a certain class given the features. If the estimated probability is greater than 50%, then the model predicts that the instance belongs to that class (positive class, labeled as 1), or else it predicts that it does not (it belongs to the negative class, labeled as 0).  

## Fairness Analysis
Finally, we will conduct a permutation test to see if our model has learned any implicit bias. We will be testing how it performs on older vs. newer recipes. We define a recipe to be 'old' if it was submitted before the median submission date in the original `recipes` DataFrame.
```python
pd.to_datetime(recipes['submitted']).median()
```
```
Timestamp('2009-05-26 00:00:00')
```

The median was in May of 2009 even though the data starts at 2008 and goes all the way through 2018. There is a very heavy skew towards early years in terms of the distribution here. We assign recipes with submission dates before then to the 'old' class and everything else is 'new'.  

**NULL HYPOTHESIS**: The best model is fair. Its F1 score is roughly the same for both old and new recipes.  
**ALTERNATIVE HYPOTHESIS**: The model is unfair and its F1 score is better for newer recipes.  

For the test statistic, we will use the signed difference in means.  (positive means the model is better for new recipes)  

We run the permutation test for just 1000 trials since the model takes a couple of seconds for inference each time, which should be enough to get a clear result.  

<iframe src="assets/f1_permtest.html" width=800 height=600 frameBorder=0></iframe>

Looks pretty significant to me! the P-value from this test is 0.0, therefore we definitely reject the null. It's possible this result has something to do with the heavy skew of dates in the data, but further analysis would be needed to conclude anything.  

* * *
### Final Thoughts

This was a very simple prediction task, but there is a lot of room to build off it and create something actually cool! We could test out different conditions for defining the *amazing* class and see how it affects model performance. We could also try out some more advanced models known to work well on tabular data like XGBoost, or even neural networks with embedding layers for the text features.  

In the future, I would like to build off of the work I started here and possibly create a recommendation engine for recipes using something like a semantic similarity search. This would be a great opportunity to learn about more advanced NLP techniques like word and sentence-level embeddings, vector databases, and approximate nearest-neighbor algorithms. Thanks for checking out my project!
