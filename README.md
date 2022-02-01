# CUNI-NU-Biocreative-Track5
Team CUNI-NU at BioCreative VII LitCovid Track: Multi-label Topical Classification of Scientific Articles using SPECTER Embeddings with Dual Attention and Label-Wise Attention Network

## Introduction
The rapid growth of biomedical literature makes manual curation and interpretation extremely difficult. During the COVID-19 pandemic, this challenge has become more apparent: the number of COVID-19-related articles in the literature is increasing at a rate of about 10,000 per month. LitCovid, a PubMed-based literature database of COVID-19-related papers, now has over 100,000 articles and receives millions of monthly accesses from users around the world.

## Data
* The training and development datasets contain the publicly-available text of over 30 thousand COVID-19-related articles and their metadata (e.g., `title`, `abstract`, `journal`). 
* Articles in both datasets have been manually reviewed and articles annotated by in-house models
* Class related statistics:

| Topic                 |  Count  |
|-----------------------|:-------:|
| epidemic forecasting  |  2048   |
| transmission          |  4409   |     
| case report           |  8696   |
| mechanism             |  18782  |
| diagnosis             |  27965  |
| treatment             |  38588  |
| prevention            |  45669  |

## Model Description

The paper describes use of a novel method of combining SPECTER embeddings, dual-attention, and a Label-wise-Attention mechanism (SPECTER-DualAtt-LWAN). The main advantage of spectre over other pretrained models is that it can be easily applied to downstream applications without the need for task-specific finetuning and it generates embeddings based on a paper’s title and abstract. Furthermore SPECTER understands bio-medical data because it has been trained on scientific corpora. These benefits of SPECTER made it a perfect choice for this task. Moreover, applying dual attention and LWAN helps model understand the interconnectivity between input sequences as well as establishing a relationship to output lables. For more information [click here](https://biocreative.bioinformatics.udel.edu/media/store/files/2021/TRACK5_pos_5_BC7_submission_188.pdf)

## Results

Below we have attached the evaluation results of proposed system on the official test data.
![image](https://user-images.githubusercontent.com/75028682/151985101-4c62b966-f21b-4b5f-856c-06772a6ca2d7.png)

## Contributors

2. Aakash Bhatnagar  (Navrachana University, Gujarat, India):  akashbharat.bhatnagar@gmail.com 
3. Nidhir Bhavsar    (Navrachana University, Gujarat, India):  nidbhavsar989@gmail.com
