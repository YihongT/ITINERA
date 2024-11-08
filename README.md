# <img src="imgs/icon.jpg" alt="icon" height="40"/> ItiNera

[[Paper]](https://arxiv.org/abs/2402.07204) [[公众号报道]](https://mp.weixin.qq.com/s/44mtENyqrHiNEEcWS61COg)

Code for our paper "ITINERA: Integrating Spatial Optimization with Large Language Models for Open-domain Urban Itinerary Planning" 

Published at **EMNLP 2024 Industry Track**

Received [**Best Paper Award**](https://raw.githubusercontent.com/YihongT/ITINERA/refs/heads/main/imgs/urbcomp.jpg) at **KDD Urban Computing Workshop (UrbComp) 2024**


## 📅 Schedule

* [x]  Release example dataset
* [x]  Release inference code


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




## 📊 Poster


![Architecture](imgs/Poster.png)


## 🛠️ Usage

Below are the instructions for running the open-source version of ItiNera. If you encounter any issues during the process, please open an issue in the repository for assistance.

### Repository Structure

The repository is organized as follows:

```
│  .gitignore
│  LICENSE
│  main.py
│  README.md
│  requirements.txt
│
├─ imgs
└─ model
    │  itinera.py
    │  itinera_en.py
    │  search.py
    │  spatial.py
    │
    ├─ data
    │
    ├─ output
    │
    └─ utils
            all_en_prompts.py
            all_prompts.py
            funcs.py
            proxy_call.py
```

### Environment Setup

This project is designed to run with **Python 3.9.20**. Required packages and dependencies are specified in `requirements.txt`.

To run ItiNera, you need to set up your API key as an environment variable. Follow these instructions to export your API key:
```
export OPENAI_API_KEY="your_api_key_here"
```

### Running Inference

To run the inference, use the following commands based on the language version:

For the Chinese version:
```bash
python main.py --type='zh'
```

For the English version:
```bash
python main.py --type='en'
```

### Example Data

Sample datasets for both the Chinese and English versions are located in the following path:
```
└─ model
    ├─ data
    │      shanghai_en.csv
    │      shanghai_en.npy
    │      shanghai_zh.csv
    │      shanghai_zh.npy
```

- **.csv files**: Represent the Points of Interest (POI) dataset, where longitude and latitude are in the GCJ-02 coordinate system.
- **.npy files**: Contain embeddings generated from the 'context' column of the POI dataset.

The above data are only used for open-source and demonstration purposes, feel free to change to your own data in actual deployments.


### Output and Visualization

Inference results are stored in `.json` files, while visualizations are provided in `.html` format. The output files are located in:

```
└─ model
    ├─ output
    │      2024_11_08_15_33_en.html                     
    │      2024_11_08_15_33_en_fulltsp.html             
    │      2024_11_08_15_33_en_response_clusters.html   
    │      2024_11_08_15_36_zh.html
    │      2024_11_08_15_36_zh_fulltsp.html
    │      2024_11_08_15_36_zh_response_clusters.html
    │      result_en.json                               
    │      result_zh.json
```

- **`<date_time>_en.html`**: Visualization file for the generated itinerary in English.
- **`<date_time>_en_fulltsp.html`**: Visualization file showing the ordered candidate Points of Interest (POIs) for the English itinerary.
- **`<date_time>_en_response_clusters.html`**: Visualization of POI clusters for the English itinerary.
- **`<date_time>_zh.html`, `<date_time>_zh_fulltsp.html`, `<date_time>_zh_response_clusters.html`**: Similar visualization files as above but in Chinese.
- **`result_en.json`**: Inference results in JSON format for the English itinerary.
- **`result_zh.json`**: Inference results in JSON format for the Chinese itinerary.

A visualization script is provided, integrating with Folium to display interactive maps.


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

This project is released under the [license](LICENSE). 
