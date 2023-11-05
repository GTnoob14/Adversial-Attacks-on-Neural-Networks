# Adversial Attacks on Neural Networks

### 1) FGSM - Fast Gradient Sign Method (WhiteBox Attack)

Method: https://arxiv.org/pdf/1412.6572.pdf

Outcome:
Original | Adversial Image
:--------:|:----------------|
![original](./fgsm-0.0.png) | ![adv](./fgsm-0.05.png)

Using different epsilon values and comparing their effectiveness
On normally trained NN | On retrained NN (with adversial examples)
:-----:|:----|
![retrained](./initial_fgsm_of_e-0.2.png) | ![normal](./retrained_fgsm_of_e-0.2.png)

Retraining the model with the 

### 2) PGD - Projected Gradient Descent (WhiteBox Attack)

Method: https://arxiv.org/pdf/1706.06083.pdf

Original | Adversial Image | Adversial Pattern
:-------:|:---------------:|:-----------------|
![original](./fgsm-0.0.png) | ![adv](./pgd_adversarial_image.png) | ![pattern](./pgd_adversial_pattern.png)

L-inf: 0.04
L2: 1.0446

### 3) ZOO - Zeroth Order Optimisation (BlackBox Attack)

Method: https://openreview.net/pdf?id=rJlk6iRqKX

Original | Adversial Images
:-----------:|:-------------|
![original](./fgsm-0.0.png) | ![adv](./bba_rgf_1.png) ![adv2](./bba_rgf_2.png)

Quality of the Black Box Adversial Images varies depending on the random initial direction-gradient.

