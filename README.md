# Potholes-Detection

### Question

坑洼道路检测和识别是一种计算机视觉任务，旨在通过数字图像（通常是地表坑洼图像）识别出存在坑洼的道路。这对于地质勘探、航天科学和自然灾害领域的研究和应用具有重要意义。例如，它可以帮助在地球轨道上识别坑洼，以及分析和模拟地球表面的形态。

在坑洼道路检测任务中，传统的分类算法往往不能取得很好的效果，因为坑洼图像的特征往往是非常复杂和多变的。然而，近年来深度学习技术的发展，为坑洼道路检测提供了新的解决方案。

深度学习具有很强的特征提取和表示能力，可以从图像中自动提取出最重要的特征。在坑洼图像分类任务中，利用深度学习可以提取到坑洼的轮廓、纹理和形态等特征， 并将其转换为更容易分类的表示形式。同时，还可以通过迁移学习和知识蒸馏等技术进一步提升分类性能。例如，一些研究者使用基于深度学习的方法对道路图像进行分类，将其分为正常、坑洼两类；另外，一些研究者还是用基于迁移学习的方法，从通用的预训练模型中学习坑洼图像的特征，并利用这些特征来分类坑洼图像。

本赛题希望通过对已标记的道路图像进行分析、特征提取与建模，从而对于一张新的道路图像能够自动识别坑洼状态。具体任务如下：

问题1：结合给出的图像文件，提取图像特征，建立一个识别率高、速度快、分类准确的模型，用于识别图像中的道路是正常或者坑洼。
问题2：对问题1中构建的模型进行训练，并从不同维度进行模型评估

#### Idea

##### 对于特征提取部分：

我们首先想到图像的一些常见且人可理解的特征：**纹理**、**梯度**、**边缘**。

* 纹理(Texture): 图像的纹理可以定义为图像的表面特征。纹理可以通过颜色、对比度和深度的变化来区分。
        
    + 纹理特征提取方法：LBP、HOG、SIFT、SURF、Gabor、Harris、Hessian、SUSAN、FAST、BRIEF、ORB、FREAK、LATCH、BRISK、AKAZE、LUCID、DAISY、PHOW、C-SIFT、C-SURF、C-HOG、C-LBP、C-WLD
    
    + 局部二进制模式 (LBP) 是一个纹理描述符，因此它通过识别每个像素与其相邻像素的强度来识别局部纹理模式。它将已标识的模式表示为二进制数字。通过计算在图像的不同区域内的这些模式的直方图，可以计算出定义局部纹理信息的特征表示。这个特性对于执行纹理分类很有用，因为它对关于图像的纹理和局部模式的信息进行编码。它的目标是以局部邻域像素值之间的关系形式来定义纹理。因此，局部二进制模式对光照条件和对比度的变化是稳健的。

    + 从LBP图像中我们可以看到，LBP识别并突出了道路上不同模式的坑洞，它基本上突出了纹理和边缘或边界的突然变化。
  
    + 这里，会存在一个问题，如果图像中有大小和坑洞形状相似或大小相近物体，如图像中的汽车，模型可能会出现误判。

    + 从图中我们可以清楚地看到，坑洞区域与汽车一起突出显示。对于二分类模型而言，因为存在坑洞区域突出，汽车不会影响模型，因为在坑洞类的主要特征/模式将仍然是坑洞
    
    + 对于分类，除了对坑洞特征的识别外，普通图像在道路上具有均匀的图案也很重要，因为在没有坑洞的情况下，道路的纹理是保持不变的  

    + 但是存在对于非坑洞图片模型误判的可能性，为了增强模型鲁棒性，消除包含汽车的图片所带来的影响，我们通过白色mask使得坑洞类的主要特征/模式是坑洞 

* 梯度(Gradient): 

* 边缘(Edge): 图像的边缘是图像中灰度变化的地方。边缘检测是图像处理中的一种基本操作，它可以检测出图像中灰度变化的地方。
    
### Pre-process

我们首先考虑提取图像的特征，然后再进行分类。我们使用了以下特征提取方法：

### Feature Fusion

融合多种特征图进行图像分类是一种常见的策略，可以有效提高模型的性能。

特征融合分为两种方式：**模型级融合**和**特征级融合**。
