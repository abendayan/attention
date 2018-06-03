##Shortcut-Stacked Sentence Encoders for Multi-Domain Inference

This is an adaptation in dynet of the model presented in https://arxiv.org/pdf/1708.02312.pdf

I used the glove.42B.300d.txt.
Download them from: http://nlp.stanford.edu/data/glove.42B.300d.zip
To run the model, I put the glove's words and the snli data in a folder named data.
If they are in an other folder, give the correct path in the config.py file.
To run, use the command:
python model.py
It is recommended to run with autobatch enabled:
python model.py --dynet-autobatch 1
