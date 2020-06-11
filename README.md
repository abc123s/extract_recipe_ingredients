# extract_recipe_ingredients
Peter Zhang's (SUNetId: pzhanggg) CS 230 Final Project

## Scripts
### Training a model
You can train a model using the command `python train.py` after setting the desired hyperparameters and dataset to train on in the file. This will train the model, and generate a new sub-directory within the `experiments` directory with the params used, performance of the model, the final model weights, and tensorboard logs.

### Fine-tuning a model
You can fine-tune a model using the command `python fine_tune.py` after specifying the experiment directory containing the model to fine tune as well the desired hyperparameters for the fine-tuning in the file. Like `python train.py` all experiment outputs will be stored in a new sub-directory of the `experiments` directory.

### Run error analysis
You can compute a confusion matrix get a sample of errors made by a model using the command `python error_analysis.py`. All you need to do is specify the experiment directory you would like to run the error analysis on in the file.

### Pre-tagging training examples for manual tagging process
You can use a model to pre-tag training examples with guessed tags in preparation for manual tagging using the command `python constructTrainingExample.py`. All you need to do is specify the experiment directory containing the model you would like to use to guess tags, as well as the file containing the training examples in `constructionTrainingExamples.py`.

This script is used in conjuction with scripts located here:
https://github.com/abc123s/example_tagger

## Tour of other files
`evaluate.py` is a helper file containing the function used to compute summary statistics for the performance of a model (word-level and sentence-level accuracy).

`build_model.py` is a helper file that builds a keras model given various hyperparameters

`masked_accuracy.py` is a custom metric to be used with `model.fit`, because the standard accuracy function does not mask padded values in the batched data.

`preprocess_*.py` are helper files to construct `tf.data.Dataset`s and encoders for the tokens and tags.

`data` is a directory containing all the data used in training the models.
