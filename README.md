# Enhancing deep discriminative feature maps via perturbation for face presentation attack detection

# Abstract
Face presentation attack detection (PAD) in unconstrained conditions is one of the key issues in face biometric-based authentication and security applications. In this paper, we propose a perturbation layer — a learnable pre-processing layer for low-level deep features — to enhance the discriminative ability of deep features in face PAD. The perturbation layer takes the deep features of a candidate layer in Convolutional Neural Network (CNN), the corresponding hand-crafted features of an input image, and produces adaptive convolutional weights for the deep features of the candidate layer. These adaptive convolutional weights determine the importance of the pixels in the deep features of the candidate layer for face PAD. The proposed perturbation layer adds very little overhead to the total trainable parameters in the model. We evaluated the proposed perturbation layer with Local Binary Patterns (LBP), with and without color information, on three publicly available face PAD databases, i.e., CASIA, Idiap Replay-Attack, and OULU-NPU databases. Our experimental results show that the introduction of the proposed perturbation layer in the CNN improved the face PAD performance, in both intra-database and cross-database scenarios. Our results also highlight the attention created by the proposed perturbation layer in the deep features and its effectiveness for face PAD in general.
<p align="center">
![alt text](https://ars.els-cdn.com/content/image/1-s2.0-S0262885619304512-gr2.jpg)
</p>

# Citations:
If you are interested in our work, please cite it as follows:

```
@article{rehman2020enhancing,
  title={Enhancing deep discriminative feature maps via perturbation for face presentation attack detection},
  author={Rehman, Yasar Abbas Ur and Po, Lai-Man and Komulainen, Jukka},
  journal={Image and Vision Computing},
  volume={94},
  pages={103858},
  year={2020},
  publisher={Elsevier}
}
```
# Collaborations:
Collaborations are welcomed, please reach out to me at yasar.abbas@my.cityu.edu.hk
