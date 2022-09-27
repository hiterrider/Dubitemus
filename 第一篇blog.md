#对几篇利用深度学习进行PolSAR图像分类文献的笔记
##1、PolSAR Image Classification Using Polarimetric-Feature-Driven Deep Convolutional Neural Network (使用特征驱动的深度CNN网络进行PolSAR图像分类)

本文于2018年4月发表于IEEE。作为一篇较早的应用CNN到PolSAR图像处理中的文章，本文主要证明了CNN网络的实用性以及利用**后向散射特性**作为输入进行学习的基本方法。选择输入的特征时不仅用到了常规的平均极化角、极化熵等，还加入了之前作者论文中提出的旋转域分解参数。作者将自主选择的特征和只采用T矩阵参量进行CNN分析做了对比，证明了使用散射机理作为先验知识得到的特征集更具有分类精度的优势。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="C:\Users\韩方舟\Desktop\Blog\第一篇blog\20220803154806.png" width = "100%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig1.本文采用的CNN结构
  	</div>
</center>

实现CNN的过程较为简单，并且是实数CNN网络。但是有问题如下：

1.**首先，极化分解产生的参数之间是否存在相关性**，也就是说通道之间的相关性是否得到了充分的表达。大多数CNN网络在处理多通道时，比如RGB图像，第一个卷积层利用卷积核移动时会简单地将在三个通道的卷积结果相加，这是否是一种信息的流失，对于PolSAR图像这种特征维数目较多时，这个问题更值得考虑，

2.**所选取的参数是否代表了最佳的参数集**，哪些参数适合用于神经网络分类，如何选择合适的参数集，参数集内部是否有冲突，是否有相关性，相关性会对分类结果产生怎样的影响，都是值得研究的内容。

##2、基于卷积神经网络的PolSAR图象精细分类方法研究
本文是硕士毕业论文。文章首次使用3D卷积神经网络来进行PolSAR图像分类。与2D不同的是，3D卷积神经网络无论在卷积层还是池化层都有针对通道维度的操作，从而尽可能地学习通道中的相关信息。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="C:\Users\韩方舟\Desktop\Blog\第一篇blog\20220803162102.png" width = "100%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig2.2D和3D卷积神经网络结构对比
  	</div>
</center>

其中一种分类结果：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="C:\Users\韩方舟\Desktop\Blog\第一篇blog\20220803162800.png" width = "100%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig3.2D和3D卷积神经网络分类对比
  	</div>
</center>

3D神经网络确实有作为分析多维度特征数据的潜力，但是不知道为什么在PolSAR近几年的研究中没有频繁出现，本文发表于2018年，如今已经有Resnet等多种分类模型，却没有出现与之相对应的3D版本，值得进一步研究。

##3、基于自动搜索多尺度CNN的PolSAR图像分类方法研究
本文是研究生毕业论文。优势在于设计了三种不同尺度的采样方法提取不同的图像特征。

**第一是图片整体作为输入直接进行全局的CNN特征提取**。以较粗的分辨率获得全局的空间特征。值得注意的是，池化层采取了**小波变换**的形式，由于Polsar图像的高频噪声易造成特征劣化，所以利用小波变换将高频部分去掉，从而达到优化的目的。

共采用了三个复数卷积层，三个复数小波池化层以及一个复数全局平均池化。
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="C:\Users\韩方舟\Desktop\Blog\第一篇blog\20220803163758.png" width = "100%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig4.复数CNN网络全局特征提取
  	</div>
</center>
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="C:\Users\韩方舟\Desktop\Blog\第一篇blog\20220803164131.png" width = "100%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig5.复数离散小波池化
  	</div>
</center>

**第二是使用1×1的复数卷积核对图像中心及其附近5×5的区域进行细粒度的特征提取**。

由于只关心中心像素周围的5×5的较小空间，因此不用池化，只需要卷积层。主要作用是提取较细粒度的高维特征。
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="C:\Users\韩方舟\Desktop\Blog\第一篇blog\20220803164650.png" width = "100%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig6.1×1复数卷积模块
  	</div>
</center>

**第三是复数Transformer模块。**

该模块和ViT提取特征的方法一样，将图片分解为25个patch，然后通过线性网络映射为1×32的tokens，完全采用transformer进行特征提取，输出的是1×32的融合向量。
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="C:\Users\韩方舟\Desktop\Blog\第一篇blog\20220803165442.png" width = "100%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig7.CV-Transformer模块
  	</div>
</center>

上述三种尺度的采样均输出1×32的特征列，再进行融合。融合的策略在实验中进行制定（后文的实验证明卷积融合的效果较好）。融合后经由Softmax函数映射为类别数。
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="C:\Users\韩方舟\Desktop\Blog\第一篇blog\20220803165601.png" width = "100%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig8.融合输出
  	</div>
</center>
其次，本文为了优化超参数的选择过程，采取了自动搜索的方法，分类结果如下。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="C:\Users\韩方舟\Desktop\Blog\第一篇blog\20220803170122.png" width = "100%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig9.旧金山地物的分类结果
  	</div>
</center>

分别采用不同的方法提取特征显然也是逻辑清晰的，因为单一的方法往往在局部和全局上无法统一。但是如果使用这种三合一的方法，应该会显著增加计算的复杂度，需要根据任务的需求进行取舍。还有一点就是，复数CNN究竟是不是必要的。