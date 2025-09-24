# CBSA (Contract-and-Broadcast Self-Attention) 

This repository is the official PyTorch implementation the paper:
+ **Towards Interpretable and Efficient Attention: Compressing All by Contracting a Few [NeurIPS 2025 | [arXiv](https://arxiv.org/abs/2509.16875)]**.

This paper extends our previous work that explored inherently interpretable Transformer decoders for semantic segmentation:
+ **Rethinking Decoders for Transformer-based Semantic Segmentation: A Compression Perspective [NeurIPS 2024 | [arXiv](https://arxiv.org/abs/2411.03033) | [github](https://github.com/QishuaiWen/DEPICT)]**

Our CBSA is an inherently interpretable and efficient self-attention mechanism that offers the following advantages:
+ It is well-established on an optimization objective grounded in the principle of compression, where the forward pass of CBSA naturally arises from its optimization procedure.
+ It scales linearly with sequence length when the number of representatives is fixed.
+ It unifies a broad spectrum of attention mechanisms as special cases, reducing their fundamental differences to variations in the number and structure of representatives.
+ It demonstrates performance comparable to, or even surpassing, linear attention while maintaining nearly identical computational cost.

<p align="center">
    <img src="assets/CBT_arch.png" width="600"\>
</p>
<p align="center">


## News
[2025/9/19] Our paper has been accepted to NeurIPS 2025 as a Spotlight🌟!

