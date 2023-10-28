# ML-ImpactorsOnInSARCoherence

Machine-Learning Characteristics of Impactors on InSAR Coherence in Wetland: Case Study in Everglade which is funded by an innovation project at Nanjing Tech University

## Motivation

1. What is InSAR coherence? Why is it important?
    InSAR coherence, or Interferometric Synthetic Aperture Radar Coherence, is a technique used in radar remote sensing that helps us understand the changes and properties of the Earth's surface. Imagine you take two radar images of the same location at different times. InSAR coherence helps us understand how similar or different these two images are. If the surface has remained relatively stable, the coherence will be high, indicating little change. However, if there have been significant movements or changes, the coherence will be lower, indicating a larger difference between the two images. 
    Extracting significant features that impact coherence products is important because it helps us identify the causes of changes in coherence. By understanding these features, we can better interpret the data and draw more accurate conclutions about what is happening on the Earth's surface. This information is vital for making informed decisions in various fields, including environmental management, disaster risk reduction, and infrastructure planning. 

2. Understanding InSAR coherence in wetland environments and quantifying their contributions remains limited.

## Goal of Project

1. Evaluation of InSAR coherence for wetland change analysis;
2. Identify significant factors affecting coherence using statistical analysis and machine learning algorithms;
3. Compare and select the optimal model and output the feature importance. 


## Data

Satellite synthetic aperture radar data, altimetry, gauge station data (EDEN) and ground truth data for environmental variables from 2016 to 2019 have been pre-processed in excel files. We also generate two excel files for multi-source datasets in dry seasons and wet seasons respectively. 

## Codes

1. Step 1: Exploring Data Analysis (EDA): Through Statistical analysis, we understand, resample, combine multi-sources datasets and prepare data to machine learning. 

2. Step 2: After EDA process, we train and conduct a regression ML model to infer feature importances on the InSAR coherence variations. We also test three different methods for hyperparameter tuning and seek optimal ML models. We compare the metrics of R2, RMSE and MAE before and after tuning.


## Current Solution 

Currently, we have processed, explored, resampled the multi-sources datasets in Everglade;
The current solution is based on Random Forest Algorithm, and three different methods: random search, grid search and bayesian search to conduct hyperparameter tuning. 


## Citation
!! If you want to use our processed datasets and python codes for your research purpose, please cit our repository or the future coming paper. If you have any questions, you can contact me or the team lead.


## Team Members
<br> Shanshan Li: shli11@outlook.com (Mentor)
<br> Bing Xu:  (Graduate Student and Secondary Mentor)
<br> Jie Yue: 2057826328@qq.com (Undergraduate Student, Team Lead)
<br> Wanxiu Zhang: (Undergraduate Student, Team Member)




