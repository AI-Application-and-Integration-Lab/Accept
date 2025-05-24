# ACCEPT: Adaptive Codebook for Composite and Efficient Prompt Tuning
[![ACL 2024 Findings](https://img.shields.io/badge/EMNLP%20Findings-2024-blueviolet)](https://aclanthology.org/2024.findings-emnlp.900/)

Official codebase for the paper, **"ACCEPT: Adaptive Codebook for Composite and Efficient Prompt Tuning"**  

## 🌟 Overview

Prompt Tuning has been a popular Parameter-Efficient Fine-Tuning method attributed to its remarkable performance with few updated parameters on various large-scale pretrained Language Models (PLMs). Traditionally, each prompt has been considered indivisible and updated independently, leading the parameters increase proportionally as prompt length grows. To address this issue, we propose Adaptive Codebook for Composite and Efficient Prompt Tuning (ACCEPT). In our method, we refer to the concept of product quantization (PQ), allowing all soft prompts to share a set of learnable codebook vectors in each subspace, with each prompt differentiated by a set of adaptive weights. We achieve the superior performance on 17 diverse natural language tasks including natural language understanding (NLU) and question answering (QA) tasks by tuning only 0.3% of parameters of the PLMs. Our approach also excels in few-shot and large model settings, highlighting its significant potential.
  <p align="center">
<img src="./image/ov2.png" width="800" alt="Comparison on datasets"/>

  </p>

<details>
<summary> 📊 Click to expand 'Performance on GLUE and SuperGLUE with T5-base model.'</summary>

  <br>

  <p align="center">
    <img src="./image/cp1.png" width="800" alt="Comparison on datasets"/>
  </p>

</details>

## 📦 Installation

## 🚀 Usage

## 📖 Citation

If you find this code useful, please cite our paper:

```
@article{lin2024accept,
  title={ACCEPT: Adaptive Codebook for Composite and Efficient Prompt Tuning},
  author={Lin, Yu-Chen and Li, Wei-Hua and Chen, Jun-Cheng and Chen, Chu-Song},
  journal={arXiv preprint arXiv:2410.12847},
  year={2024}
}
```


