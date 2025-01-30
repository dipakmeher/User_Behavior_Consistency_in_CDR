import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import csv
import argparse

hf_token = '<YOUR HUGGINGFACE TOKEN>'
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load the tokenizer and model using the Hugging Face token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=hf_token
)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

# Set up argument parsing
parser = argparse.ArgumentParser(description="Read and process a CSV file containing Amazon reviews.")
parser.add_argument("--csv_file_path", type=str, help="Path to the input CSV file")
parser.add_argument("--output_file_path", type=str, help="Path to save the output CSV file")
args = parser.parse_args()
csv_file_path = args.csv_file_path
output_file_path = args.output_file_path

# Read data from CSV file
data = pd.read_csv(csv_file_path)

# Function to generate response
def generate_response(input_ids, model, tokenizer, min_length=150, max_attempts=3):
    for attempt in range(max_attempts):
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = tokenizer.convert_tokens_to_ids("</s>")

        outputs = model.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
        #repetition_penalty=1.2
        response = outputs[0][input_ids.shape[-1]:]
        generated_text = tokenizer.decode(response, skip_special_tokens=True)

        if len(generated_text.split()) >= min_length:
            return generated_text

    return None  # Return None if all attempts fail to meet the minimum length

# Function to check if response is valid
def is_response_valid(response, min_length=150):
    # Check if the response meets the minimum length
    if len(response.split()) < min_length:
        return False
    # Additional checks for validity can be added here
    return True

# Iterate over each row in the CSV file and process the data
results = []
for index, row in data.iterrows():
    user_id = row['book_user_id']
    book_asin = row['book_asin']
    movie_asin = row['movie_asin']
    book_rating = row['book_rating']
    movie_rating = row['movie_rating']

    # Adding the new variables for book and movie data
    book_generated_response = row['book_generated_response']
    book_sentiment_response = row['book_sentiment_response']
    movie_generated_response = row['movie_generated_response']
    movie_sentiment_response = row['movie_sentiment_response']
   
    
    # BOOKS MOVIES DOMAIN
    messages = [
    {"role": "system", "content": "You are an advanced AI model that rigorously determines the consistency between user behaviors based on their preferences and interactions across different domains. Focus on clear, meaningful alignments in behavior, considering observable patterns such as thematic focus, narrative complexity, and user engagement across domains like books and movies. Avoid abstract or unrelated inferences and ignore emotional tone or sentiment. Base conclusions solely on specific, measurable patterns in user behavior."},

    {"role": "user", "content": f"""Below are the item details for one item in each domain the user has interacted with:

**For the Book Domain:**
- **Book Item:** {book_generated_response}

**For the Movie Domain:**
- **Movie Item:** {movie_generated_response}

Based on the item details from the two domains (Books and Movies), assess the consistency of the user's behavior across these domains. Identify clear connections between preferences, thematic interests, and engagement, focusing on patterns in user behavior. Consider consistent preferences for themes, complexity, or genres, even when expressed through different mediums.

When assessing consistency, use the following guidelines:
1. Highly-consistent: The user's behavior shows clear and meaningful alignment across both domains, with significant similarities in preferences or behavior patterns that are relevant, even if these are subtle or abstract. This alignment goes beyond simply liking or disliking both items.

2. Neutral: The user's behavior shows some similarities, but there is no strong or clear pattern of consistency or divergence across the domains.

3. Not-consistent: The user's behavior is noticeably inconsistent across both domains, with clear differences in preferences or behavior patterns, and no significant alignment is observed in how they engage with the content across the two domains.

Please classify the user's behavior as one of the following: 'Highly-consistent,' 'Neutral,' or 'Not-consistent.' Provide a brief explanation of your decision.
"""}
]

  
    
    '''
    #BOOK FOOD DOMAIN    
    messages = [
    {"role": "system", "content": "You are an advanced AI model that rigorously determines the consistency between user behaviors based on their responses, preferences, and sentiments across different domains. Your task is to critically assess how consistently the user interacts with similar content in different domains. Recognize that the correlation may be subtle due to the inherent differences between domains, but ensure that only clear and meaningful alignments in their behavior and sentiments are identified as consistent. Avoid drawing abstract or forced conclusions based solely on matching sentiments (whether positive or negative)."},
    
    {"role": "user", "content": f"""Below are the item details and corresponding sentiments for one item in each domain the user has interacted with:

**For the Book Domain:**
- **Book Item:** {book_generated_response}
- **Sentiment for the Book:** {book_sentiment_response}

**For the Food Domain:**
- **Food Item:** {movie_generated_response}
- **Sentiment for the Food:** {movie_sentiment_response}

Based on the item details and sentiments provided for each item in the two domains (books and food), critically assess the consistency of the user's behavior when interacting with similar content across these domains. Consider any potential behavioral patterns or correlations in how the user engages with each item, but **do not base consistency solely on matching positive or negative sentiments**. Ensure that your evaluation is based on clear patterns that can be meaningfully compared across domains.

Choose one of the following options:

1. **Highly-consistent:** The user's behavior shows clear and meaningful alignment across both domains, with significant similarities in preferences or behavior patterns that are relevant, even if these are subtle or abstract. This alignment goes beyond simply liking or disliking both items.
   
2. **Neutral:** The user's behavior shows some similarities, but there is no strong or clear pattern of consistency or divergence across the domains.

3. **Not-consistent:** The user's behavior is noticeably inconsistent across both domains, with clear differences in preferences or behavior patterns, and no significant alignment is observed in how they engage with the content across the two domains."""
}
]
'''
    '''
    #MOVIE MUSIC DOMAIN
    messages = [
    {"role": "system", "content": "You are an advanced AI model that rigorously determines the consistency between user behaviors based on their responses, preferences, and sentiments across different domains. Your task is to critically assess how consistently the user interacts with similar content in different domains, focusing on meaningful alignments in their behavior and sentiments. Ensure that only clear, strong patterns of alignment across domains are considered consistent, and avoid giving too much weight to abstract or minor similarities."},

    {"role": "user", "content": f"""Below are the item details and corresponding sentiments for one item in each domain the user has interacted with:

**For the Movie Domain:**
- **Movie Item:** {book_generated_response}
- **Sentiment for the Movie:** {book_sentiment_response}

**For the Music Domain:**
- **Music Item:** {movie_generated_response}
- **Sentiment for the Music:** {movie_sentiment_response}

Based on the item details and sentiments provided for each item in the two domains (movies and music), critically assess the consistency of the user's behavior when interacting with similar content across these domains.

For **Handling of Subtle Correlations**:
   When subtle correlations are present, prioritize broad behavioral patterns, such as the user’s general engagement with emotional depth, thematic complexity, or sensory elements, over isolated details. Subtle patterns should only be identified as consistent if they strongly align with the user’s behavior across both domains. Avoid classifying abstract or surface-level similarities as consistent behavior unless they are part of a broader trend.

When assessing consistency, use the following guidelines:
1. Highly-consistent: The user's behavior shows clear and meaningful alignment across both domains, with significant similarities in preferences or behavior patterns that are relevant, even if these are subtle or abstract. This alignment goes beyond simply liking or disliking both items.

2. Neutral: The user's behavior shows some similarities, but there is no strong or clear pattern of consistency or divergence across the domains.

3. Not-consistent: The user's behavior is noticeably inconsistent across both domains, with clear differences in preferences or behavior patterns, and no significant alignment is observed in how they engage with the content across the two domains.

Please classify the user's behavior as one of the following: 'Highly-consistent,' 'Neutral,' or 'Not-consistent.' Provide a brief explanation of your decision.
 
   """}
]
''' 
    '''   
    #ELECTRONIC FOOD DOMAIN
    messages = [
    {"role": "system", "content": "You are an advanced AI model that rigorously determines the consistency between user behaviors based on their responses, preferences, and sentiments across different domains. Your task is to critically assess how consistently the user interacts with similar content in different domains, recognizing that the correlation may be subtle due to the inherent differences between domains. Focus on broader patterns such as the user’s attention to detail, practicality, or emotional satisfaction, and ensure that only clear and meaningful alignments in their behavior and sentiments are identified as consistent. Avoid relying solely on liking or disliking items."},

    {"role": "user", "content": f"""Below are the item details and corresponding sentiments for one item in each domain the user has interacted with:

**For the Electronics Domain:**
- **Electronics Item:** {book_generated_response}
- **Sentiment for the Electronics:** {book_sentiment_response}

**For the Food Domain:**
- **Food Item:** {movie_generated_response}
- **Sentiment for the Food:** {movie_sentiment_response}

Based on the item details and sentiments provided for each item in the two domains (electronics and food), critically assess the consistency of the user's behavior when interacting with similar content across these domains. Recognize that **functional differences** between electronics (e.g., usability, design, features) and food (e.g., taste, health, freshness) may lead to different types of engagement. However, focus on broader behavioral patterns.

When assessing consistency, use the following guidelines:
1. **Highly-consistent**: The user's behavior shows clear and meaningful alignment across both domains, with significant similarities in preferences or behavior patterns that are relevant, even if these are subtle or abstract. This alignment goes beyond simply liking or disliking both items.

2. **Neutral**: The user's behavior shows some similarities, but there is no strong or clear pattern of consistency or divergence across the domains.

3. **Not-consistent**: The user's behavior is noticeably inconsistent across both domains, with clear differences in preferences or behavior patterns, and no significant alignment is observed in how they engage with the content across the two domains.

Please classify the user's behavior as one of the following: 'Highly-consistent,' 'Neutral,' or 'Not-consistent.' Provide a brief explanation of your decision.
"""}
]
    '''


    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    generated_text = generate_response(input_ids, model, tokenizer, min_length=1)

    if generated_text and is_response_valid(generated_text, min_length=1):
        results.append({
            "user_id": user_id,
            "user_correlation": generated_text,
            "book_asin": book_asin,
            "movie_asin": movie_asin,
            "book_rating": book_rating,
            "movie_rating": movie_rating,
            "Book_Generated_Response": book_generated_response,
            "Book_Sentiment_Response": book_sentiment_response,
            "Movie_Generated_Response": movie_generated_response,
            "Movie_Sentiment_Response": movie_sentiment_response,
        })
    else:
        results.append({
            "user_id": user_id,
            "user_correlation": "Failed to generate a valid response",
            "book_asin": book_asin,
            "movie_asin": movie_asin,
            "book_rating": book_rating,
            "movie_rating": movie_rating,
            "Book_Generated_Response": book_generated_response,
            "Book_Sentiment_Response": book_sentiment_response,
            "Movie_Generated_Response": movie_generated_response,
            "Movie_Sentiment_Response": movie_sentiment_response,
        })

    # Define the correct fieldnames that match the keys in the result dictionaries
    fieldnames = [
        "user_id", 
        "user_correlation", 
        "book_asin", 
        "movie_asin", 
        "book_rating", 
        "movie_rating", 
        "Book_Generated_Response", 
        "Book_Sentiment_Response", 
        "Movie_Generated_Response", 
        "Movie_Sentiment_Response"
    ]

# Save the responses to a CSV file
with open(output_file_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f"Responses saved to {output_file_path}")

