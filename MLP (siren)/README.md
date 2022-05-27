## Compression using NLP

This method is based on SIREN paper (https://www.vincentsitzmann.com/siren/). We used the SIREN model to predict pixel value given three coordinates: index of the projection and the position in the projection.
Other approach is to split the projections into several parts and to train several SIREN models on each part. This lead to faster training, higher quality and lower compression rate.

This folder contains python notebook, which can be also found in colab: 
To reproduce result just click 'run all cells'. You can also tune hyperparameters in the last cell of the notebook. 
Set `model_type='siren'` for vanilla SIREN model. Set `model_type='siren_cascade'` for a collection of SIREN models, each of one will train on its own part on the data.