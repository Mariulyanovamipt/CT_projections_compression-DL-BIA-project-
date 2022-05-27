# CT_projections_compression-DL-BIA-project-
This repository contains code implementation of different compression method which we have implemented and compared on CT projections dataset as a part of our joint DL and BIA project at Skoltech.

# Project description
In this project we implemented, compare and combine several compression ideas.
We work with raw projections (images caught by CT detector), which then are used for human body reconstruction. Our goal is to compress projection storage.

We have implemented the folowing methods:
1) VQ-VAE image compression applied to each image
2) Deep video compression
3) ConvLSTM frame prediction
4) MLP neural image represemtation
5) Using less number of projections, but getting the same quality on reconstruction (UNet architecture)

Related code coud be found in corresponding folder

