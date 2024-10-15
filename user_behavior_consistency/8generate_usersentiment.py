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
    # Check if the response is None
    if response is None:
        return False
    
    # Check if the response meets the minimum length
    if len(response.split()) < min_length:
        return False
    # Additional checks for validity can be added here
    return True

# Iterate over each row in the CSV file and process the data
results = []
for index, row in data.iterrows():
    user_id = row['user_id']
    asin = row['asin']
    rating = row['rating']
    title = row['title']
    description = row['description']
    features = row['features']
    review = row['text']
    rating = row['rating']
   
    
    # FOR BOOKS MOVIES DOMAIN
    messages = [
    {"role": "system", "content": "You are an advanced AI model that extracts user sentiment from reviews based on given details. Your task is to determine the overall sentiment expressed by the user, taking into account both the review text and the rating provided. Please consider specific nuances relevant to both the Book and Movie domains."},
    {"role": "user", "content": f"""Title: {title}
Description: {description}
Features: {features}
Review: {review}
Rating: {rating}

Focus on the following aspects:
1. **User Sentiment**: Identify the overall sentiment expressed by the user in the review. The sentiment can be 'Positive', 'Neutral', or 'Negative'.

2. **Rating Context**: Consider the rating given (e.g., 4 out of 5) as a strong indicator of sentiment. High ratings (4 or 5) generally correspond with a positive sentiment unless the review text suggests otherwise. Pay attention to whether the rating aligns with the tone of the review.

3. **Handling Conflicts and Mixed Sentiments**: If the review contains both positive and negative elements, or if there is a conflict between the review text and the rating, focus on the **overall rating** and the **general tone** of the review to determine the dominant sentiment. Always aim for consistency in applying this approach across all reviews.

4. **Review Tone and Domain Relevance**: Consider the tone of the review, focusing on elements relevant to the specific domain:
    - **For Books**: Focus on elements like storytelling, character development, writing style, plot depth, pacing, and emotional impact.
    - **For Movies**: Focus on elements like plot, acting, direction, visual effects, pacing, and cinematography.

5. **Domain-Specific Sentiment Indicators**:
    - **For Books**: Positive sentiment may include appreciation for storytelling, plot depth, character arcs, or writing style, while negative sentiment may involve criticism of character development, pacing, or storyline.
    - **For Movies**: Positive sentiment may include praise for the plot, acting, direction, or visual effects, while negative sentiment may involve criticism of pacing, acting, or story coherence.

Please provide your answer as a single word ('Positive', 'Neutral', or 'Negative') followed by a brief explanation (one or two sentences) behind your decision, incorporating the relevant domain-specific aspects.
"""
    }
]
    
    '''
    #BOOK FOOD DOMAIN    
    messages = [
    {"role": "system", "content": "You are an advanced AI model that extracts user sentiment from reviews based on given details. Your task is to determine the overall sentiment expressed by the user, taking into account both the review text and the rating provided. Please consider specific nuances relevant to both the Book and Food domains."},
    
    {"role": "user", "content": f"""Title: {title}
Description: {description}
Features: {features}
Review: {review}
Rating: {rating}

Focus on the following aspects:
1. **User Sentiment**: Identify the overall sentiment expressed by the user in the review. The sentiment can be 'Positive', 'Neutral', or 'Negative'.

2. **Rating Context**: Consider the rating given (e.g., 4 out of 5) as a strong indicator of sentiment. High ratings (4 or 5) generally correspond with a positive sentiment unless the review text suggests otherwise. Pay attention to whether the rating is in line with the overall tone of the review.

3. **Handling Mixed or Conflicting Sentiments**: If the review contains both positive and negative elements, focus on the **overall rating** and the **general tone** of the review to identify the dominant sentiment. If the review is unclear, prioritize the rating as the more reliable indicator of the user's sentiment. However, if the review clearly expresses strong emotions (positive or negative) that contradict the rating, prioritize the dominant sentiment from the review.

4. **Review Tone and Domain Relevance**: Consider the tone of the review, focusing on aspects relevant to the specific domain:
   - **For Food**: Focus on elements like taste, freshness, packaging, and satisfaction.
   - **For Books**: Focus on elements like storytelling, character development, writing style, and emotional impact.

5. **Domain-Specific Sentiment Indicators**: For food products, positive sentiment may include praise for taste, quality, and freshness, while negative sentiment may involve complaints about flavor, texture, or packaging. For books, positive sentiment may include appreciation for the plot, characters, or writing style, while negative sentiment may involve criticism of the storyline, pacing, or character development.

**Please provide your answer as a single word ('Positive', 'Neutral', or 'Negative') followed by a brief explanation (one or two sentences) behind your decision, incorporating the relevant domain-specific aspects. If there is a conflict between the rating and review tone, explain which aspect (rating or review tone) was prioritized and why.**
"""}
]
 '''
    '''     
    #FOR MOVIE MUSIC DOMAIN
    messages = [
    {"role": "system", "content": "You are an advanced AI model that extracts user sentiment from reviews based on given details. Your task is to determine the overall sentiment expressed by the user, taking into account both the review text and the rating provided. Please consider specific nuances relevant to both the Movie and Music domains."},
    
    {"role": "user", "content": f"""Title: {title}
Description: {description}
Features: {features}
Review: {review}
Rating: {rating}

Focus on the following aspects, ensuring that the sentiment analysis remains grounded in the domain-specific elements as well as any relevant information inferred from the item data:

1. **User Sentiment**: Identify the overall sentiment expressed by the user in the review. The sentiment can be 'Positive', 'Neutral', or 'Negative'.

2. **Rating Context**: Consider the rating given (e.g., 4 out of 5) as a strong indicator of sentiment. High ratings (4 or 5) generally correspond with a positive sentiment unless the review text suggests otherwise. Pay attention to whether the rating is in line with the overall tone of the review.

3. **Handling Mixed or Conflicting Sentiments**: If the review contains both positive and negative elements, focus on the **overall rating** and the **general tone** of the review to identify the dominant sentiment. If the review is unclear, prioritize the rating as the more reliable indicator of the user's sentiment. However, if the review clearly expresses strong emotions (positive or negative) that contradict the rating, prioritize the dominant sentiment from the review.

4. **Review Tone and Domain Relevance**: Consider the tone of the review, focusing on elements that match the domain and align with the points identified below:
   - **For Movies**: Focus on theme/emotional tone, cultural/contextual significance, narrative/storytelling elements, and pacing.
   - **For Music**: Focus on theme/emotional tone, cultural/contextual significance, rhythm/flow, and aesthetic style.

5. **Domain-Specific Sentiment Indicators**:
   - **For Movies**: Positive sentiment may include appreciation for the emotional tone, cultural relevance, narrative depth, or pacing, while negative sentiment may involve criticism of plot coherence, pacing, or narrative engagement.
   - **For Music**: Positive sentiment may include appreciation for the emotional tone, rhythm, cultural significance, and production quality, while negative sentiment may involve criticism of the melody, production, or overall flow.

**Please provide your answer as a single word ('Positive', 'Neutral', or 'Negative') followed by a brief explanation (one or two sentences) behind your decision, incorporating the relevant domain-specific aspects and any important information from the item data.**
"""}
]
    '''
    
    '''
    #FOR ELECTRONICS FOOD DOMAIN
    messages = [
    {"role": "system", "content": "You are an advanced AI model that extracts user sentiment from reviews based on given details. Your task is to determine the overall sentiment expressed by the user, taking into account both the review text and the rating provided. Please consider specific nuances relevant to both the Electronics and Food domains. If domain-specific elements are not mentioned, infer sentiment based on the overall tone of the review and the rating."},
    {"role": "user", "content": f"""Title: {title}
Description: {description}
Features: {features}
Review: {review}
Rating: {rating}

Focus on the following aspects:
1. **User Sentiment**: Identify the overall sentiment expressed by the user in the review. The sentiment can be 'Positive', 'Neutral', or 'Negative'.
2. **Rating Context**: Consider the rating given (e.g., 4 out of 5) as a strong indicator of sentiment. High ratings (4 or 5) generally correspond with a positive sentiment unless the review text suggests otherwise. Pay attention to whether the rating is in line with the overall tone of the review.
3. **Handling Mixed or Conflicting Sentiments**: If the review contains both positive and negative elements, focus on the **overall rating** and the **general tone** of the review to identify the dominant sentiment. If the review is unclear, prioritize the rating as the more reliable indicator of the user's sentiment. However, if the review clearly expresses strong emotions (positive or negative) that contradict the rating, prioritize the dominant sentiment from the review.

4. **Review Tone and Domain Relevance**: If specific elements related to the domain are mentioned, consider the following:
    - **For Food**: Focus on elements like taste, freshness, packaging, and satisfaction.
    - **For Electronics**: Focus on elements like performance, durability, ease of use, design, and technical specifications.
    If no specific domain elements are mentioned, infer sentiment based on the general tone of the review text and rating.
5. **Domain-Specific Sentiment Indicators**: For food products, positive sentiment may include praise for taste, quality, and freshness, while negative sentiment may involve complaints about flavor, texture, or packaging. For electronics, positive sentiment may include satisfaction with performance, design, and usability, while negative sentiment may involve issues with functionality, build quality, or ease of use.

Please provide your answer as a single word ('Positive', 'Neutral', or 'Negative') followed by a brief explanation (one or two sentences) behind your decision, incorporating the relevant domain-specific aspects or the overall tone of the review text and rating if no specific elements are mentioned."""}
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
            "asin": asin,
            "rating": rating,
            "Title": title,
            "Description": description,
            "Features": features,
            "Review": review,
            "Generated_Response": generated_text
        })
    else:
        results.append({
            "user_id": user_id,
            "asin": asin,
            "rating": rating,
            "Title": title,
            "Description": description,
            "Features": features,
            "Review": review,
            "Generated_Response": "Failed to generate a valid response"
        })

# Save the responses to a CSV file
with open(output_file_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["user_id", "asin", "rating", "Title", "Description", "Features", "Review", "Generated_Response"])
    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f"Responses saved to {output_file_path}")

