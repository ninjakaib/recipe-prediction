# Savor or Skip
## The Art of Recipe Rating Prediction
Using Machine Learning to find out what's worth eating


## Problem Identification
In this project, we will continue to build off the progress we made analyzing the recipes and reviews dataset from before. This time, we will explore building a predictive model to classify recipes. The classification task we are going to attempt is determining which recipes are *amazing!*  

We could try predicting the rating in terms of the number of stars, but since most of the reviews are 5 stars, we could easily achieve an accuracy of around 70% just by picking 5 every time. We have to choose some condition that recipes must satisy to be considered *amazing.*

**1. Recipes with only 5 star reviews**: This condition will include recipes that are generally well-received, but could potentially include recipes that have been reviewed only once (thus, there might not be a consensus on their quality).

**2. Recipes with only 5 star reviews and more than one review**: This condition is more stringent and could exclude some good recipes that received a single non-perfect rating.

**3. Recipes that have been reviewed more than once, where every review gave a rating of at least 4 stars, and at least one review awarded the maximum 5 stars**: This seems like a good balance. It ensures that the recipe is consistently highly-rated but doesn't require perfection in all reviews.

**4. Only recipes that have more than one review and then apply the condition of recipes with only 5 star reviews**: This condition attempts to find a balance by excluding recipes with only one review (which might not be representative) and then looking for consistently perfect scores. These will be the same recipes as in condition 2, but with a smaller overall dataset.

The condition we choose depends on what we want are model to be able to classify. We can think of lower proportions corresponding to 'higher standards' and the classifier will learn to identify 'better' recipes. For this project, the classifier should be trying to find the very best recipes, so we will go with condition 2. This will help us find out what sets those top recipes apart. Other conditions can be tried later to see how it affects models.  

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
|   recipe_id |   minutes |   n_steps |   n_ingredients | ingredients                                                                                                                            | nutrition                                        |   rating |   count |   mean |   is_amazing |
|------------:|----------:|----------:|----------------:|:---------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------|---------:|--------:|-------:|-------------:|
|      275022 |        50 |        11 |               7 | ['cheddar cheese', 'macaroni', 'milk', 'eggs', 'bisquick', 'salt', 'red pepper sauce']                                                 | [386.1, 34.0, 7.0, 24.0, 41.0, 62.0, 8.0]        |        1 |       3 |      3 |            0 |
|      275026 |        45 |         7 |               9 | ['frozen crabmeat', 'sharp cheddar cheese', 'cream cheese', 'onion', 'milk', 'bisquick', 'eggs', 'salt', 'nutmeg']                     | [326.6, 30.0, 12.0, 27.0, 37.0, 51.0, 5.0]       |        1 |       2 |      3 |            0 |
|      275030 |        45 |        11 |               9 | ['apple pie filling', 'graham cracker crust', 'cream cheese', 'sugar', 'vanilla', 'eggs', 'caramel topping', 'pecan halves', 'pecans'] | [577.7, 53.0, 149.0, 19.0, 14.0, 67.0, 21.0]     |        5 |      10 |      5 |            1 |
|      275035 |         5 |         5 |               7 | ['honey', 'peanut butter', 'powdered milk', 'chocolate chips', 'raisins', 'coconut', 'nuts']                                           | [1908.7, 152.0, 801.0, 44.0, 133.0, 174.0, 71.0] |        4 |       2 |      4 |            0 |
|      275036 |        15 |         6 |               8 | ['frozen corn', 'green pepper', 'onion', 'garlic cloves', 'butter', 'olive oil', 'salt & pepper', 'parmesan cheese']                   | [270.4, 20.0, 4.0, 3.0, 11.0, 37.0, 12.0]        |        5 |       2 |      5 |            1 |


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

Predicting based on these features will force our model to (hopefully) learn about what intrinsically makes a recipe good rather than using the user review text. We will quantify how good our model is with F1 score. F1 is a combination of recall and precision; recall is important especially for the positive class since we want the model to be able to identify a high proportion of the *amazing* recipes, and precision is also important since we don't want the model to achieve high recall by simply guessing positive for everything. F1 is a good balance of these two metrics and also better than accuracy especially if we decide to use a different condition for *amazingness* that would make the classes less evenly distributed, similar to how they are in the original dataset. For example, if we vary the condition just slightly to include recipes with only 5 star reviews, but drop the condition that all recipes must have more than a single review, the positive class would only make up 24% of our data. This could lead to a relatively high accuracy but the model won't actually be good at predicting what we want.

## Baseline Model

The baseline model will be a simple model that doesn't use all of the features we selected to give us a ... you guessed it ... baseline of performance to compare future models against. To keep things simple, we won't worry about the nutrition column until later. 


## Final Model



## Fairness Analysis

<iframe src="assets/f1_permtest.html" width=800 height=600 frameBorder=0></iframe>
