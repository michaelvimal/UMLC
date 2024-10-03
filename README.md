## The UMLC Dataset and Comprehensive Model Evaluation

This project focuses on implementing and evaluating four state-of-the-art models — Graph2Tree, BertGen, GPT-2, and GPT-3 — to solve math word problems. These models are designed to interpret natural language math problems, generate corresponding equations, and predict accurate answers. The project leverages pre-trained models and custom deep learning architectures for tackling the complexities of math word problem-solving.

## Key Components:
1. **Graph2Tree**: A sequence-to-tree model, implemented via **MWPToolkit**, that generates math equations from word problems.
2. **BertGen**: A sequence generation model based on **BERT**, used to generate math solutions by leveraging bi-directional attention.
3. **GPT-2**: An autoregressive model used for generating text-based math solutions through pre-training and fine-tuning.
4. **GPT-3**: A large-scale language model accessed via OpenAI's API to generate high-quality solutions to math word problems.

#### Workflow:
- **Data Processing**: Datasets like **UMLC**, **MATH**, and **GSM8K** are preprocessed using tokenizers and customized datasets for model training.
- **Model Training & Evaluation**: The models are trained and evaluated on these datasets, with performance metrics like **Accuracy**, **F1 Score**, and **Loss** being calculated.
- **Performance Comparison**: The effectiveness of each model is compared across multiple datasets, providing insights into their strengths and weaknesses in solving math word problems.

This project highlights the application of advanced NLP models for educational tasks, showcasing the intersection of language understanding and mathematical reasoning.


## Installtion Requirements
Read theRrequirements.txt file to understand the basic components required for implementing this project.

## Directory structure

UMLC_Project/

│
├── datasets/         # Contains datasets and file tuned copies(UMLC, MATH, GSM8K)

├── Docs/             # Project installtion documentation

├── Logs/             # Stores logs for model training and evaluation and results

├── Models/           # Contains model scripts (Graph2Tree, BertGen, GPT-2, GPT-3)

├── README.md         # Overview or description of the project

└── Requirements.txt  # List of required Python packages and dependencies


For deployments, you have to change the project path according to your project requirements. You may need to have different versions of python environment to implement and execute the models. Its advisable to have a conda platform to run each model indepentenly. 


### UMLC Dataset Overview:

The UMLC dataset is a specialized dataset developed for solving math word problems using machine learning models. It contains complex questions from various fields of mathematics, with each problem paired with a corresponding equation or solution.

### Key Features:
1. Question-Answer Format: 
   Each entry consists of a math word problem (question) and its solution (answer), covering topics from basic arithmetic to advanced algebra and beyond.

2. Application: 
   The dataset is used to train models like Graph2Tree, BertGen, GPT-2, and GPT-3, supporting sequence-to-sequence tasks to translate natural language math problems into accurate solutions.

3. Structure: 
   Stored in JSON format, the dataset is easy to integrate into machine learning workflows.

In this study, we compare the performance of benchmark models across three different datasets, including UMLC, MATH, and GSM8K, using metrics like accuracy, loss, and F1 score. The UMLC dataset serves as a valuable resource for advancing model capabilities in solving diverse math word problems.



### Evaluation metric

We have implemented evaluation metrics such as Accuracy, Loss and F1 Score to measure the effect of MWP models.  


## Experiment Results

We have implemented the models on the datasets that are integrated within our toolkit. All the implementation follows the build-in configurations. All the experiments are conducted with 5 cross-validation. The experiment results(Equ acc|Val acc) are displayed in the following table.

# UMLC
