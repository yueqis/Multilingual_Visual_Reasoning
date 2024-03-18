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

3. Run the experiments: we experiment various models, including GPT-4V, LLaVA, mBLIP, CCLM, UNITERs, and ViLT. Each model requires different environment, so we include a `README.md` file for each model in its directory, indicating how to reproduce the experiments. 

## Experiments

Here, we briefly describe all experiments we run. More details can be found in our paper. We first evaluate models' performance on NLVR2 and MaRVL under zero-shot or finetuned settings, and propose three interventions: translate test, visual programming, and reasoning with captions.

### Evaluation

#### Zero-Shot

In this setting, models are not specifically fine-tuned for the task of visual reasoning. Instead, we use LMM's zero-shot learning abilities by providing them prompts with instructions on how to solve the task. We evaluate the following models:

GPT-4V: Instructions on how to reproduce the results are provided in [gpt4v/README.md](https://github.com/yueqis/Multilingual_Visual_Reasoning/blob/main/gpt4v/README.md).

LLaVA: Instructions on how to reproduce the results are provided in [LLaVA/README.md](https://github.com/yueqis/Multilingual_Visual_Reasoning/blob/main/LLaVA/README.md).

mBLIP: Instructions on how to reproduce the results are provided in [mBLIP/README.md](https://github.com/yueqis/Multilingual_Visual_Reasoning/blob/main/mBLIP/README.md).

#### Finetuned

In this setting, models are finetuned on the English NLVR2 dataset, and test them on both NLVR2 and MaRVL. We evaluate the following models:

mBLIP: Instructions on how to reproduce the results are provided in [mBLIP/READEME.md](https://github.com/yueqis/Multilingual_Visual_Reasoning/blob/main/mBLIP/README.md).

CCLM: Instructions on how to reproduce the results are provided in [CCLM/READEME.md](https://github.com/yueqis/Multilingual_Visual_Reasoning/blob/main/CCLM/README.md).

UNITERs: We evaluate xUNITER and mUNITER, and instructions on how to reproduce the results are provided in [uniters/READEME.md](https://github.com/yueqis/Multilingual_Visual_Reasoning/blob/main/uniters/README.md).

### Interventions

We perform three interventions. The pipelines for each can be found in the flow chart below.

<img src="images/interventions.png"
alt="Intervention Pipelines" width="600"/> 

#### Translate Test

Instead of evaluating models on multilingual text, we evaluate models on the MaRVL dataset with multilingual text translated to English using Google translate API.

#### Visual Programming

We use the approach introduced in [VisProg](https://github.com/allenai/visprog), solving multimodal reasoning using LLMs to generate visual programs, that leverage off-the-shelf computer vision models for image processing during inference.

#### Reasoning with Captions

We first caption both images, and use an LLM to reason about the given statement with the two captions, rather than with the two images.

### Zeno Result Visualization

We upload all our experimental results to [Zeno](https://zenoml.com/), a platform which we use to visualize our results. An example visualization is shown below.

<img src="images/zeno.png"
alt="Example of Zeno" width="600"/> 

Note: Our paper is under anonymous review. Since the links to zeno is unanonymized, we will post the links to zeno post-review. 

### TODO

Our paper is under anonymous review. We will post the citation to our paper post-review. 
