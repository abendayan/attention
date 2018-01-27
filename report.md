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
**86,3%** on test

I did not succeed in replicating the result.

## Performance on the dataset
* Train:
* Test

#### Graphs

## Attempts to replicate the result
After reading the paper, I understood that the model has a simple architecture:
>Embedding => (projection) => F => calculate score => concatenate => G => result => sum => concatenate => H => final layer

F, G, and H being feed forwards networks, F and G have a RelU activation function.

## What worked, what didn't work
I made an error in calculating the score after the layer F (I used the probability instead of the score for the second sentence), in consequence of that, the loss and the accuracy converged very slowly.

I had a lot of issue with the memory. The glove words used here (and in most of the papers) use a lot of memory, so I switched to the dependencies words from word2vec.

The model is simple enough that I did not have issue in connecting the layers.

## Improvements
The score go through multiple networks, I'm wondering if it might be better to save the score and to concatenate it with the result from the final layer.
