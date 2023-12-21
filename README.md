# Knowledge Tracing: Comparing DNNs Versus GPT For Intelligent Tutoring Systems

This repository contains the files for the second project of EPFL ML4Science Machine Learning course (CS-433).

## Team Members

- Seyed Parsa Neshaei <seyed.neshaei@epfl.ch>
- Adam Hazimeh <adam.hazimeh@epfl.ch>
- Bojan Lazarevski <bojan.lazarevski@epfl.ch>

## Hosting Lab

This project was done in collaboration and under the supervision of Dr. Richard Lee Davis from the CHILI lab at EPFL (led by Prof. Pierre Dillenbourg).

## Project Abstract

Knowledge tracing (KT) and predicting the next answers of students is considered important in designing intelligent tutoring systems. Previous studies have utilized deep neural networks (DNNs) for this task. However, recent research shows large language models (LLMs) may emerge prediction capabilities by prompting. Previous studies have not compared the performance of DNNs with LLMs on KT. In this project, we evaluate the performance of three types of DNNs for recurrent data, against GPT-3.5, on a set of real-world exercises from a database course. We find a lower performance than DNNs, suggesting limited indications of the LLMs' abilities in KT.

## Installation and Reproducablity

We provide two requirements files for the project. The first one is `requirements-dnn.txt` which installs packages required for running the DNN models and fine-tuning GPT-2. The second one is `requirements-gpt-35.txt` which installs packages required for inference using GPT-3.5 API. The files are automatically extracted using the `conda list -e` command. Use `conda create --name <env> --file [name of the relevant requirements file]` to create an environment with the required packages. To re-run the code, first run the `data_processor.ipynb` notebook to generate the required JSON files. Then, run the `DNN Knowledge Tracing.ipynb` notebook to train the DNN models. Finally, run the `GPT Knowledge Tracing.ipynb` notebook to run inference using GPT-3.5 API using the different set of requirements. The parameters for models can be modified in each file respectively. Additionally, the `GPT-2 Fine-tuning.ipynb` file contains the code for fine-tuning GPT-2 on the dataset. However, we did not use this model in our experiments in the main report as it did not perform well; we only included it as a complimentary experiment in the Appendix of the report. If you receive any error related to `accelerate` or `transformers`, running `pip install -U accelerate` and then `pip install -U transformers` may solve the issue.

## Project Structure

The project is structured as follows:

- `data_outputs` directory: Contains the output JSON files from the data preperation code, acting as inputs to the model training code. For more details, see `data_outputs/README.md`.

- `dataverse_files` directory: Contains the raw data from the dataset we used.

- `data_processor.ipynb` notebook: Contains the code for data preperation and extracting disjoint subsequences.

- `DCE.pdf`: Contains the filled digital ethics canvas.

- `DNN Knowledge Tracing.ipynb` notebook: Contains the code for training the DNN models.

- `GPT Knowledge Tracing.ipynb` notebook: Contains the code for running inference using GPT-3.5 API.

- `gpt_helpers.py`: Contains helper functions for running inference using GPT-3.5 API in the `GPT Knowledge Tracing.ipynb` notebook.

- `GPT-2 Fine-tuning.ipynb` notebook: Contains the code for fine-tuning GPT-2 on the dataset.

- `report.pdf`: Contains the final report of the project.

- `requirements-dnn.txt`: Contains the list of packages required for running the DNN models and fine-tuning GPT-2.

- `requirements-gpt-35.txt`: Contains the list of packages required for running inference using GPT-3.5 API.

- `seq_helpers.py`: Contains helper functions for the sequence DNN models in the `DNN Knowledge Tracing.ipynb` notebook.

The description of all code is added either as markdown or as code comments in the files.



