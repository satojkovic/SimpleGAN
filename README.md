# SimpleGAN

Original implementation is provided with this link. 
[https://towardsdatascience.com/demystifying-generative-adversarial-networks-c076d8db8f44](https://towardsdatascience.com/demystifying-generative-adversarial-networks-c076d8db8f44)


## Setup

* Datasets
  * [zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)

```csh
$ git clone https://github.com/satojkovic/SimpleGAN
$ cd SimpleGAN
$ wget https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/utils/mnist_reader.py
$ python train.py
```

## Results

* Epoch 1
  ![example1](results/gan_generated_image_epoch_1.png)

* Epoch 100
  ![example2](results/gan_generated_image_epoch_100.png)

* Epoch 300
  ![example3](results/gan_generated_image_epoch_300.png)
