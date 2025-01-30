# Understanding User Behavior Consistency in Cross-Domain Recommendation: An LLM-Based Approach

Cross-domain recommendation (CDR) has recently emerged as a promising solution to address the cold-start and sparsity issues faced by single-domain recommender systems. Users often exhibit different interests and rating behaviors across various domains. For instance, a user may rate items in the Movies domain differently than in the Books domain. In this work, we empirically analyze whether state-of-the-art CDR algorithms make significantly better recommendations in a target domain when a user’s rating behavior is consistent across domains. We present a novel approach that uses Large Language Models (LLMs) to quantify a consistency value, measuring how consistent a user’s behavior is in rating items across different domains. Our empirical analysis reveals that while state-of-the-art CDR models perform better on domain pairs with high user behavior consistency, they often lack statistically significant improvements in recommendation accuracy across diverse domain combinations. These findings highlight the need for new directions in CDR algorithm design that better leverage user behavior consistency to enhance recommendation performance. 


# Environment - Installations of Libraries and Packages
Create new python environment and Install below packages. 
python -m venv cdr_env
source cdr_env/bin/activate 

#Update pip
pip install --upgrade pip

#Install required Python packages

#Hugging Face Transformers for AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BertModel
pip install transformers

#Hugging Face Tokenizers
pip install tokenizers

#PyTorch
pip install torch

#Pandas for data manipulation
pip install pandas

#Tqdm for progress bars
pip install tqdm

#NumPy for numerical operations
pip install numpy

#Scikit-learn for normalization and cosine similarity
pip install scikit-learn

pip install accelerate

#CSV, argparse, and os are part of Python's standard library, so no need to install those.

#Add Hugging Face token to the environment
echo "Please enter your Hugging Face token:"
read hf_token
huggingface-cli login --token "$hf_token"

#Done
echo "All required packages have been installed and Hugging Face token added."

# Data Preparation
The data preparation scripts for both the proposed method and the baselines are located in the datasets folder. The files are numbered in the sequence they should be run. Each script takes inputs from the terminal.

# User Behavior Consistency
The code for the proposed User Behavior Consistency method is located in the user_behavior_consistency folder. The method is modularized, with each module having its own file. The files are numbered to indicate the order they should be executed.

For example:
1generate_preferences2_minlengthcheck.py should be run first.
12user_correlationscore.py should be run at the end.
All scripts take input from the terminal. Below is an example command to run the first file:

python 1generate_preferences2_minlengthcheck.py \
       --csv_file_path ../dataset/Amazon_Reviews23/data_results/6_3book.csv   \
       --output_file_path ./result/1_3bookmovie_book_up_wus_withoutsentiment_minlengthcheck50.csv

Key Prompt Files:
Item Feature Extraction: File number 1
Sentiment Extraction: File number 8
User Behavior Extraction: File number 10

# Baselines
The code for baseline models can be found in the baselines folder. To execute the baseline models, please refer to the official documentation of the PTUPCDR source code.

# Conclusion
This project examines the importance of user behavior consistency in cross-domain recommendation. We hope the proposed LLM-based method and findings will inspire future improvements in CDR models. Follow the structured file sequence and input requirements outlined here to replicate and extend this research.

For any issues or inquiries, feel free to contact us.


      
