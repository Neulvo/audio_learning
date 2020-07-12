# How to Improve GAN model

reference :  https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b



## problems

1. Non-Convergence : never ending cat and mouse game

2. Mode collapse

3. slow training



## ways to improve GAN

1. change the cost function
2. additional penalties to the cost function
3. avoid overconfidence and overfitting
4. better optimization
5. add label



### Feature Matching

- changes the cost function for the generator

  => minimizing the statistical difference between the features of the real images and the generated images

  ![Image for post](https://miro.medium.com/max/1600/1*7ZM4HlUE81WyvxXZVhprrw.png)

  ![Image for post](https://miro.medium.com/max/1756/1*ad8tUoJx7U3tfMDRcs0MGg.jpeg)

### Minibatch discrimination

- model collapses -> all images created looks similar

  => we feed real images and generated images into the discriminator separately in different batches

  & compute the similarity of the image *x* with images in the same batch

  => We append the similarity ***o(x)\*** in one of the dense layers in the discriminator to classify whether this image is real or generated.

![Image for post](https://miro.medium.com/max/1760/1*Bv5alWimwu3DxiNDpFJhpg.jpeg)

![Image for post](https://miro.medium.com/max/2000/1*c5mlMuqp1CZQ_UlQDV7c6Q.jpeg)



### One-sided label smoothing

- overconfidence problem

  => penalize the discriminator when the prediction for any real images go beyond 0.9 (*D(real image)>0.9*)



``p=tf.placeholder(tf.float32, shape=[None,10])``

``feed_dict={p:[[0,0,0,0.9,0,0,0,0,0,0]]}`` # image with label "3"

``d_real_loss=tf.nn.sigmoid_cross_entropy _with_logits(labels=p, logits=logits_real_image)``



### Historical averaging

- In historical averaging, we keep track of the model parameters for the last t models

- Alternatively, we update a running average of the model parameters if we need to keep a long sequence of models

  => add and L2 cost below to the cost function to penalize model different from the historical average

![Image for post](https://miro.medium.com/max/1412/1*GPF-atOVcSTbXN6C8ALqKg.png)

### Experience replay

- model optimization can be too greedy in defeating what the generator is currently generating

   => Instead of fitting the models with current generated images only, we feed the discriminator with all recent generated images also.

  => discriminator will not be overfitted for a particular time instance of the generator.



### Using labels(CGAN)

- Adding the label as part of the latent space ***z\*** helps the GAN training. 

![Image for post](https://miro.medium.com/max/1600/1*CVnxcMtCLendyrnLvOlKzA.png)

### Cost functions

![Image for post](https://miro.medium.com/max/1242/1*sE-ChIllxdrzIQBQhi33UQ.jpeg)

table refer : https://github.com/hwalsuklee/tensorflow-generative-model-collections

WGAN/WGAN-GP : https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490

EBGAN/BEGAN : https://medium.com/@jonathan_hui/gan-energy-based-gan-ebgan-boundary-equilibrium-gan-began-4662cceb7824

LSGAN : https://medium.com/@jonathan_hui/gan-lsgan-how-to-be-a-good-helper-62ff52dd3578

RGAN/RaGAN : https://medium.com/@jonathan_hui/gan-rsgan-ragan-a-new-generation-of-cost-function-84c5374d3c6e

=> need further study

below : FID score(Fréchet Inception Distance (FID score))

- FID is a measure of similarity between two datasets of images. 

![Image for post](https://miro.medium.com/max/1062/1*814M7wjVMl18Ma4uTIbecQ.jpeg)

further research : https://arxiv.org/pdf/1711.10337.pdf

tunning hyperparameters also important

- different learning rates for the generator and the discriminator.

  further reserach : https://arxiv.org/pdf/1706.08500.pdf



### Implementation tips

- Scale the image pixel value between -1 and 1. Use *tanh* as the output layer for the generator
- Experiment sampling **z ** with Gaussian distributions
- Batch normalization often satbilizes training
- Use <u>PixelShuffle</u> and <u>transpose convolution</u> for upsampling
- Avoid max pooling for downsampling. <u>Use convolution stride.</u>
- **Adam** optimizer usually works better than other methods

- <u>Add noise</u> to the real and generated images before feeding them into the discriminator



### Virtual batch normalization(VBN)

- BM(Batch Normalization) can create a dependency between samples

  -> The generated images are not independent of each other

  => To mitigate that, we can combine a reference batch with the current batch to compute the normalization parameters.

![Image for post](https://miro.medium.com/max/1336/1*tvIXKHtObzRKVZbT97aO5A.png)

further research : https://arxiv.org/pdf/1701.00160v3.pdf

### Random seeds

-  initialize the model parameters impact the performance of GAN.
- further research : https://arxiv.org/pdf/1711.10337.pdf



### Batch normalization

- DGCAN strongly recommends adding BM into the network design.
- The use of BM also become a general practice in many deep network model.
- However, there will be exceptions. (as seen at the WGAN GP)

![Image for post](https://miro.medium.com/max/978/1*Ttfns14b8Cfxg3NcmVzsow.jpeg)

source :  https://arxiv.org/pdf/1711.10337.pdf

### Spectral Normalization

- weight normalization that stabilizes the training of the discriminator.

-  It controls the Lipschitz constant of the discriminator 

  => to mitigate the exploding gradient problem

  => the mode collapse problem.

- Lipschitz [continuous function](https://en.wikipedia.org/wiki/Continuous_function) is limited in how fast it can change

![d_{Y}(f(x_{1}),f(x_{2}))\leq Kd_{X}(x_{1},x_{2}).](https://wikimedia.org/api/rest_v1/media/math/render/svg/a7b2d5f533b6c4c8c63ce33c85d8763e4eaffbfd)

​       => Any such *K* is referred to as **a Lipschitz constant** for the function *f*. 



### Multiple GANs

- we may collect the best model for each mode and use them to recreate different modes of images.

![Image for post](https://miro.medium.com/max/1600/1*LDDGart2Ri5fRvxqp1c-9A.png)

source : https://arxiv.org/pdf/1611.02163.pdf

- In case need further research



### Balance between discriminator & generator

- Balancing both networks with dynamic mechanics is also proposed. But not until recent years, we get some traction on it.
- some researchers challenge the feasibility and desirability of balancing these networks.
  - A well-trained discriminator gives quality feedback to the generator anyway. 
  -  Also, it is not easy to train the generator to always catch up with the discriminator.
  -  Instead, we may turn the attention into finding a cost function that does not have a close-to-zero gradient when the generator is not performing well.

![Image for post](https://miro.medium.com/max/1517/1*6So6q3dWurG8qrmwk1y3jw.jpeg)

### Discriminator & generator network capacity

The model for the discriminator is usually more complex than the generator (more filters and more layers) and a good discriminator gives quality information. 



### BigGAN

#### **Larger batch size**

![Image for post](https://miro.medium.com/max/1066/1*sKfPbx8Jb94ZIsRG-pNjFw.png)

The smaller the FID score, the better

source : https://arxiv.org/pdf/1809.11096.pdf

- Increase the batch size have a significant drop in FID
- BigGAN reports the model reaches better performance in fewer iterations, but become unstable and even collapse afterward.

#### Truncation Trick

- Low probability density region in the latent space *z* may not have enough training data to learn it accurately.
- So when generating images, we can avoid those regions to improve the image quality at the cost of the variation.
- i.e. the quality of images will increase but those generated images will have lower variance in style. There are different techniques to truncate the input latent space *z*.
- The general principle is when values fall outside a range, it will be resampled or squeeze to the higher-probability region.

#### Increase model capacity

- During tuning, consider increasing the capacity of the model, in particular for layers with high-spatial resolutions. 

- But don’t do it too early without proofing the model design and implementation first.

#### Moving averages of Generator weights

#### **Orthogonal regularization**

- A matrix *Q* is orthogonal if

![Image for post](https://miro.medium.com/max/1600/0*tCHKA9dwRoINDy8m.jpeg)

- If we multiply *x* with an orthogonal matrix, the changes in *x* will not be magnified. This behavior is very desirable for maintaining numerical stability.

![Image for post](https://miro.medium.com/max/1300/0*lCpPKac4zB4_4VyI.png)

- We can add an orthogonal regularization to encourage such properties during training. It penalizes the system if Q deviates from being an orthogonal matrix.

  ![Image for post](https://miro.medium.com/max/1600/1*cUKuOmY1SaANJCmdZa4grQ.jpeg)

- Nevertheless, this is known to be too limiting and therefore BigGAN uses a modified term:

![Image for post](https://miro.medium.com/max/1612/1*zey5nfDL-ZUr0tMzjW2RyA.jpeg)

       - - The Frobenius norm, sometimes also called the Euclidean norm (a term unfortunately also used for the vector ![L^2](https://mathworld.wolfram.com/images/equations/FrobeniusNorm/Inline1.gif)-norm), is [matrix norm](https://mathworld.wolfram.com/MatrixNorm.html) of an ![m×n](https://mathworld.wolfram.com/images/equations/FrobeniusNorm/Inline2.gif) matrix ![A](https://mathworld.wolfram.com/images/equations/FrobeniusNorm/Inline3.gif) defined as the [square root](https://mathworld.wolfram.com/SquareRoot.html) of the sum of the absolute squares of its elements,

![ ||A||_F=sqrt(sum_(i=1)^msum_(j=1)^n|a_(ij)|^2) ](https://mathworld.wolfram.com/images/equations/FrobeniusNorm/NumberedEquation1.gif)

- The Frobenius norm can also be considered as a [vector norm](https://mathworld.wolfram.com/VectorNorm.html).

  It is also equal to the [square root](https://mathworld.wolfram.com/SquareRoot.html) of the [matrix trace](https://mathworld.wolfram.com/MatrixTrace.html) of ![AA^(H)](https://mathworld.wolfram.com/images/equations/FrobeniusNorm/Inline4.gif), where ![A^(H)](https://mathworld.wolfram.com/images/equations/FrobeniusNorm/Inline5.gif) is the [conjugate transpose](https://mathworld.wolfram.com/ConjugateTranspose.html), i.e.,

![ ||A||_F=sqrt(Tr(AA^(H))). ](https://mathworld.wolfram.com/images/equations/FrobeniusNorm/NumberedEquation2.gif)

#### **Orthogonal weight initialization**

#### **Skip-z connection**

- In the vanilla GAN, the latent factor *z* is input to the first layer only. With skip-z connection, direct skip connections (skip-z) from the latent factor *z* is connected to multiple layers of the generator rather than just the first layer.