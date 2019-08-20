# brain-data-with-age
brain data with age in Hongik University

Master's thesis-- Age Classification From MRI Data Using RNN 2019.08

We use a deep learning network to analyze images of the human brain to estimate of subject's age.
Using functional magnetic resonance imaging (fMRI) data of the human brain,
we use Gated Recurrent Unit (GRU) deep learning neural network to explore the relationship between the age and the structure human brain.


我们使用深度学习网络来分析人类大脑的图像，以估计受试者的年龄。
使用人脑的功能磁共振成像（fMRI）数据，采用门控递归单元（GRU）深度学习神经网络来探索年龄与人类大脑结构之间的关系。

우리는 대상자의 나이를 추정하기 위해 인간 두뇌의 이미지를 분석해서 심층 학습 네트워크를 사용한다.
우리는 인간의 뇌의 기능 자기공명영상(fMRI) 데이터를 이용하여 GRU(Gated Recurrent Unit) 심층학습 신경망을 이용하여 나이와 인간의 뇌 구조 사이의 관계를 탐구한다.

First, the obtained fMRI data is processed, and the image is processed using fsl (fsl_in_linux.py)

Secondly, the processed image is compared with the preset image to obtain data of each brain domain with time (comparision.py). 

Third, the age is classified using a deep learning network (ML_GRU_all_train_test.py). 

Fourth, improve the learning network to improve training results (ML_upgrade.py).
