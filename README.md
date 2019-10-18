# brain-data-with-age
brain data with age in Hongik University

Master's thesis-- Age Classification From MRI Data Using RNN 2019.08

We use a deep learning network to analyze images of the human brain to estimate of subject's age.
Using functional magnetic resonance imaging (fMRI) data of the human brain,
we use Gated Recurrent Unit (GRU) deep learning neural network to explore the relationship between the age and the structure human brain.

First, the obtained fMRI data is processed, and the image is processed using fsl (fsl_in_linux.py)

Secondly, the processed image is compared with the preset image to obtain data of each brain domain with time (comparision.py). 

Third, the age is estimated using a deep learning network (brain_.py). 
