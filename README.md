# MulticlassReviewClassification

## Preview

In short, this repository shows my steps in solving the task in several notebooks.

|  Notebook  |  Short Description  |
| --- | --- |
|  [EDA.ipynb](https://github.com/koren-v/MulticlassReviewClassification/blob/master/EDA.ipynb)  |  Includes some Data Analysis and attempts to find a correlation between the target and non-text fields.  |
|  [Two_Berts.ipynb](https://github.com/koren-v/MulticlassReviewClassification/blob/master/Two_Berts.ipynb)  | In this notebook I decided to train two separate models on subsample and save their predictions to use them as metafeatures for the next model. As Bert has goog enough vocabulary my text was just lowercased (by default) but wasn't been cleaned. Аs the main metric, I chose F1 score as the dataset is unbalanced.  |
|  [Ensemble.ipynb](https://github.com/koren-v/MulticlassReviewClassification/blob/master/Ensemble.ipynb)  |  Shows creating an ensembles:  simple voting and blending using KNN. This approaches gave same scores compearing with single models.  |
|  [Bert_for_Pair.ipynb](https://github.com/koren-v/MulticlassReviewClassification/blob/master/Bert_for_Pair.ipynb)  |  In this notebook I tried to put into Bert pairs of sentences separating them by special token as it was done during pre-training Bert. As a result, this model showed a much better score even than an ensemble. Except for the performance, one more benefit of this approach is that we don't need to save two or even three models for evaluating.  |

## Data

The dataset was really big and as time is limited I decided that it would be a smart decision to use a subsample of data. As I mentioned before dataset was unbalanced, so to get good representative training and evaluation sets I used stratified subsampling. During EDA I found that field 'summary' has values like 'Five Stars', 'Four Stars' and so on. It looks like default value, but not each target corresponds to this value for rows with these values. It possible to use this fact as metafeature, but I decided to not implement this idea.
Also, I didn't find a linear correlation with the target in features as 'image', 'style', 'vote' and 'verified', so I didn't build the model on these features as I wanted.

<p align="center">
  <img src="/images/correlation.PNG">
</p>

## Model

The model's architecture consists of pre-trained Bert as an encoder and fully-connected head with dropout for classification. I used bert-base-uncased configuration from pytorch-pretrained-bert.

## Hyperparameters tuning

As models that I used were big (even as I used not large configuration), the training loop was really long, so I almost didn't change hyperparameters. The only thing is, I tried a different number of neighbors in KNN and made the conclusion that increasing this number we make our model more robust to overfitting that is pretty logical. 

## Comparison

|  Model  |  Validation F1-score  |
| --- | --- |
|  Bert trained on 'reviewText' field  |  0.7728  |
|  Bert trained on 'summary' field  |  0.7041  |
|  Averaging of separate Berts  |  0.7660  |
|  Blending using KNN as final model  |  0.7716  |
|  Bert with two sentences as one input  |  0.8465  |

 
