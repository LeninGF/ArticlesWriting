
- date: 2025-04-10 
- Model: Gemini 2.0 Flash
- URL:
  https://gemini.google.com/app/207a25c8e5f02425?redirect=home&hl=en
- Mode: Deep Research

You are PhD research assistant and your task is to help me to write the literature review section for the paper BERT-Based Fine-Tuning for Automated Tagging of Robbery Crime Narratives. Next I describe the subsections and what I expect for each of the subsections of the literature review. Also I provide the abstract in order to have a context about the research and the results. Please use academic tone and use only academic references in the machine learning and computer science domain. I provide some initial references which I think you should consider.



- Introduction: take into account the abstract to look for information

- Legal natural Language Processing: what is legal AI, what are the problems we try to solve with Legal Natural Language Processing. Focus specially on Text classification and Question Answering. THis section is conceptual

- BERT Transformer: this section is conceptual about the most important characteristics of BERT transformer

- Related works on Legal Text Classification: provide a comparative analysis on solutions in the art for legal text classification. what do they solve? what do they use e.g. transformers. Are there related documents focused on Robbery?

Abstract: Accurate classification of crime narratives is essential for

 generating reliable public safety statistics. In Ecuador, the

 Comisión Especial de Estadística de Seguridad, Justicia, Crimen y

 Transparencia (CEESJCT) manually categorizes robbery incident

 reports, a process that is both time-consuming and prone to human

 error. While transformer-based models have revolutionized natural

 language processing, particularly in English, their application for

 Spanish in legal and security related texts, such as those

 associated with the classification of robbery, remains

 underexplored. This study seeks to address this challenge by

 developing a machine learning model that automates the

 classification of crime reports pertaining to robbery, leveraging

 the contextual strengths of transformer architectures to address

 linguistic and domain-specific challenges and improving overall

 accuracy and efficiency. The primary objective of this study was to

 automate the classification of robbery narratives into standardized

 crime categories. For this purpose, a BERT model built on

 transformer architecture was trained using a tailored database of

 narratives and corresponding labels. The approach began with

 transfer learning to establish a solid baseline, and was further

 refined through fine tuning. Ultimately, the model achieved improved

 performance by utilizing a larger dataset; each stage contributed to

 notable enhancements. Model evaluation was strengthened by close

 collaboration with key Ecuadorian institutions, namely the Fiscalía

 General del Estado (FGE) and the Instituto Nacional de Estadística y

 Censos (INEC), whose cooperation proved pivotal in ensuring the

 model's accuracy and reliability.The baseline transfer learning

 model achieved moderate accuracy (80.5\%) but struggled with

 semantically overlapping categories, such as distinguishing

 \textit{Robo a Domicilio} from \textit{Robo a Unidades

  Económicas}. Fine-tuning resolved many of these issues, improving

 minority-class recall by up to 30\% and enabling real-time

 predictions via a Flask interface. The final scaled model

 demonstrated high robustness (95.5\% accuracy) on 11 categories,

 with cross-validation confirming consistent performance across

 police and judicial narratives.



references to consider for concepts:

- https://www.mdpi.com/2078-2489/16/2/130/htm https://www.mdpi.com/2078-2489/16/2/130

- https://arxiv.org/abs/2410.21306v2

- 10.1109/ICCCNT61001.2024.10725043

- https://aclanthology.org/2020.acl-main.466/

- https://arxiv.org/abs/2302.12039v1
