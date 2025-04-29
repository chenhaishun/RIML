# RIML

This repository contains the code of our AAAI 2025 paper "Biased Incomplete Multi-View Learning"

# Background

Existing Incomplete multi-view learning often assume that the view-missing patterns are uniformly random in multi-view data, which does not agree with real-world scenarios. In practice, view-missing patterns often vary across different classes. For example, in the medical field, patients with rare diseases would take more examinations than those with common diseases; in the financial field, high-risk customers tend to receive evaluations from more views than ordinary ones. This observation indicates that the pattern of missing views is intrinsically tied to the specific disease type, suggesting that missing patterns are class-dependent rather than uniformly random.

![image-20250429134447592](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora202504/29/134447-689476.png)

# Motivations

Previous methods would typically fail due to such biased view-missing patterns. This motivates us to delve into a new biased incomplete multi-view learning problem. To this end, we develop a Reliable Incomplete Multi-view Learning (RIML) method. RIML is a simple yet effective learning-free imputation framework that goes beyond the conventional approaches by considering information from all classes, rather than just relying on individual views or within-class samples.

# Requirements

This program has been tested in the following environments: 

* Ubuntu 18.04
* Python 3.9.16
* Pytorch 1.13.1
* scikit-learn 0.22.1
* numpy 1.26.2

# Quick Start

Download link of the story dataset can be obtained from the original paper (cited in our paper).

For training and evaluating:

```
python main_hand.py
```

<br>