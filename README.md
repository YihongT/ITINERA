# <img src="imgs/icon.jpg" alt="icon" height="40"/> ItiNera

[[Paper]](https://arxiv.org/abs/2402.07204) [[公众号报道]](https://mp.weixin.qq.com/s/44mtENyqrHiNEEcWS61COg)

Code for our paper "ITINERA: Integrating Spatial Optimization with Large Language Models for Open-domain Urban Itinerary Planning" 

Published at **EMNLP 2024 Industry Track**

Received [**Best Paper Award**](https://raw.githubusercontent.com/YihongT/ITINERA/refs/heads/main/imgs/urbcomp.jpg) at **KDD Urban Computing Workshop (UrbComp) 2024**

**We will release our code in the next few weeks**

## ⭐️ Highlights

**TL;DR:** We present ItiNera, a system that integrates spatial optimization with large language models to generate customized and efficient itineraries for the Open-domain Urban Itinerary Planning (OUIP) problem.

* Addresses personalized itinerary planning by decomposing user requests and optimizing routes using spatial clusters.
* Generates urban travel plans by selecting and organizing points of interest (POIs) based on user needs in natural language.
* Outperforms traditional methods in delivering custom and spatially efficient itineraries, validated by experiments on real-world data.



<p align="center">
<img src="imgs/ouip.jpg" alt="ouip" width="80%"/> 
</p>








## 📌 Abstract

Citywalk, a recently popular form of urban travel, requires genuine personalization and understanding of fine-grained requests compared to traditional itinerary planning. In this paper, we introduce the novel task of Open-domain Urban Itinerary Planning (OUIP), which generates personalized urban itineraries from user requests in natural language. We then present ITINERA, an OUIP system that integrates spatial optimization with large language models to provide customized urban itineraries based on user needs. This involves decomposing user requests, selecting candidate points of interest (POIs), ordering the POIs based on cluster-aware spatial optimization, and generating the itinerary. Experiments on real-world datasets and the performance of the deployed system demonstrate our system's capacity to deliver personalized and spatially coherent itineraries compared to current solutions.

<p align="center">
<img src="imgs/qualitative.jpg" alt="qualitative" width="60%"/> 
</p>



## 🔍 Method

![Architecture](imgs/architecture.jpg)



## 🛠️ Usage

- TODO

  


## 📅 Schedule

* [ ]  Release example dataset
* [ ]  Release inference code



## 🖊️ Citation

If you find this work helpful for your research, please consider giving this repo a star ⭐ and citing our paper:

```bibtex
@article{tang2024itinera,
  title={ITINERA: Integrating Spatial Optimization with Large Language Models for Open-domain Urban Itinerary Planning},
  author={Tang, Yihong and Wang, Zhaokai and Qu, Ao and Yan, Yihao and Wu, Zhaofeng and Zhuang, Dingyi and Kai, Jushi and Hou, Kebing and Guo, Xiaotong and Zhao, Jinhua and others},
  journal={arXiv preprint arXiv:2402.07204},
  year={2024}
}
```



## 📃 License

This project is released under the [MIT license](LICENSE). 
