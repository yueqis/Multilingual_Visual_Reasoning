# Multilingual Visual Reasoning

This repo includes the implementations described in the paper "*What Is Missing in Multilingual Visual Reasoning and How to Fix It*". 

## Brief Overview

In this work, we explore various models, including GPT-4V, LLaVA, mBLIP, CCLM, and UNITERs on NLVR2 and MaRVL. NLVR2 and MaRVL involves the same task of reasoning whether a natural language statement is true based on a pair of images. An example is shown below.

<img src="images/example.png"
alt="Example of Zeno" width="600"/> 

Analyzing the failures of these models, we find that multilinguality, complex reasoning, and multimodality are three key aspects that make this task so challenging to models. Based on our analysis, we explore and propose three interventions to address these three challenges: translation, visual programming, and reasoning with captions. Our interventions achieve the best open performance on this task in a zero-shot setting, boosting open model LLaVA's performance by 13.4%, while also minorly improving GPT-4V's performance.

## Repo Setup

1. Clone this repo.

2. Download the data: all json files are already in `data/`, but you will need to download the images. For NLVR2 images (around 15G), please fill out the [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdB_OhgmpQULV17kjQ4iitftILbOJjuGgJ2ECmg-HdmkjUSAg/viewform) by the [NLVR2 group](https://lil.nlp.cornell.edu/nlvr/). MaRVL images (around 2.5G) can be downloaded from the [Dataverse portal](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/42VZ4P). 

3. 

## Experiments

### Evaluation

#### Zero-Shot

#### Finetuned

We use NLVR2's train split to finetune the models, and then evaluate on MaRVL and NLVR2 dev split.

UNITERs: We evaluate xUNITER and mUNITER, and instructions on how to reproduce the results are provided in [uniters/READEME.md](https://github.com/yueqis/Multilingual_Visual_Reasoning/blob/main/uniters/README.md).

### Interventions

<img src="images/interventions.png"
alt="Intervention Pipelines" width="600"/> 

#### Translate Test

#### VisProg

#### Reasoning with Captions

### Zeno Result Visualization

We upload all our experimental results to [Zeno](https://zenoml.com/), a platform which we use to visualize our results. An example visualization is shown below.

<img src="images/zeno.png"
alt="Example of Zeno" width="600"/> 

Note: Our paper is under anonymous review. Since the links to zeno is unanonymized, we will post the links to zeno post-review. 

### TODO

Our paper is under anonymous review. We will post the citation to our paper post-review. 
