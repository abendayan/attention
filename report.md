# A Decomposable Attention Model for Natural Language Inference
> Adele Bendayan 336141056

## Choice of paper
This is an adaptation in dynet of the model presented in https://arxiv.org/pdf/1606.01933v1.pdf

## Reasons for the choice
Most of the papers that I read implemented a complicated architecture with a lot of parameters. My computer does not have a lot of memory, so I choose this one because this model has a relative simple architecture and doesn't use a lot of parameters.

## Method used in the paper
* Attend (align the 2 sentences)
* Compare (compare each sub phrase)
* Aggregate (aggregate the set and use the result to predict the label)

## Result reported in the paper
**86,3%** on test (version without inference)

I did not succeed in replicating the result.

## Performance on the dataset
* Train: 68,7%
* Test: 69,8%

#### Graphs
Accuracy on train:

![Accuracy](train_accuracy.png)

Accuracy on train:

![alt text](test_accuracy.png)

Loss on train:

![alt text](loss_train.png)

## Attempts to replicate the result
After reading the paper, I understood that the model has a simple architecture:
>Embedding => (projection) => F => calculate score => concatenate => G => result => sum => concatenate => H => final layer

F, G, and H being feed forwards networks, F and G have a RelU activation function.

SGD with learning rate of 0.1: got to 65% accuracy. Then I looked into the Tensor Flow implementation of this model, and I saw that they used Adagrad with a learning rate of 0,05, so I tried that and I got a result that is a bit better.

## What didn't work
I made an error in calculating the score after the layer F (I used the probability instead of the score for the second sentence), in consequence of that, the loss and the accuracy converged very slowly.

The glove vectors alone takes most of my computer memory, so I used only the vectors that are seen on the train set. But it means that I loss some information with the test set (words that are on the test set but are not in the train set are encoded with the UNK embedding word). This is certainly one of the reasons why the performance on my code is so much lower than theirs.

## What worked
The model is simple enough that I did not have issue in connecting the layers.

## Improvements
The score go through multiple networks, I'm wondering if it might be better to save the score and to concatenate it with the result from the final layer.
