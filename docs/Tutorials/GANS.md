## 生成对抗网络
生成对抗网络(GANS)是一个强大的概率模型方法(I. Goodfellow, 2016; I. Goodfellow et al., 2014)。它们确定一个深度生成的模型，并且它们支持快速准确的推理。

我们使用Edward的一个实例来演示。可以使用可交互的[Jupyter notebook](http://nbviewer.jupyter.org/github/blei-lab/edward/blob/master/notebooks/gan.ipynb)
```python
M = 128  # batch size during training
d = 100  # latent dimension

DATA_DIR = "data/mnist"
IMG_DIR = "img"
```
### 生成数据
我们使用MNIST的数据集，它包括55000张28*28像素的图片(LeCun, Bottou, Bengio, & Haffner, 1998)。每张图片被表示为包含784个元素的二维向量，每个元素是一个在0和1之间取值的像素点。
![](./gan-fig0.png)
我们的目标是建立并推理出一个可以生成高质量的手写数字图片的模型。
我们使用多批次MNIST数字图片进行训练。我们使用固定批量大小为M的图片集实例化一个TensorFlow变量。
```python
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
x_ph = tf.placeholder(tf.float32, [M, 784])
```

### 评价与检验

### 模型

### 推理过程

### References
Goodfellow, I. (2016). NIPS 2016 Tutorial: Generative Adversarial Networks. ArXiv Preprint ArXiv:1611.06953.

Goodfellow, I. J. (2014). On distinguishability criteria for estimating generative models. In ICLR workshop.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., … Bengio, Y. (2014). Generative adversarial nets. In Neural information processing systems.

Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A kernel two-sample test. The Journal of Machine Learning Research, 13, 723–773.

Gutmann, M. U., Dutta, R., Kaski, S., & Corander, J. (2014). Statistical Inference of Intractable Generative Models via Classification. ArXiv Preprint ArXiv:1407.4981.

Gutmann, M., & Hyvärinen, A. (2010). Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In Artificial intelligence and statistics.

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278–2324.

Marin, J.-M., Pudlo, P., Robert, C. P., & Ryder, R. J. (2012). Approximate Bayesian computational methods. Statistics and Computing, 22(6), 1167–1180.

Mohamed, S., & Lakshminarayanan, B. (2016). Learning in Implicit Generative Models. ArXiv Preprint ArXiv:1610.03483.

Rubin, D. B. (1984). Bayesianly justifiable and relevant frequency calculations for the applied statistician. The Annals of Statistics, 12(4), 1151–1172.

Sugiyama, M., Suzuki, T., & Kanamori, T. (2012). Density-ratio matching under the Bregman divergence: a unified framework of density-ratio estimation. Annals of the Institute of Statistical ….
