# VAE-ResNet18-PyTorch
A Variational Autoencoder based on the ResNet18-architecture, implemented in PyTorch.

Out of the box, it works on 64x64 3-channel input, but can easily be changed to 32x32 and/or n-channel input.  

Instead of transposed convolutions, it uses a combination of upsampling and convolutions, as described here:  
https://distill.pub/2016/deconv-checkerboard/

The implementation of the encoder is inspired by https://github.com/kuangliu/pytorch-cifar

