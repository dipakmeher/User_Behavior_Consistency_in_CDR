import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import csv
import argparse

hf_token = '<YOUR HUGGINGFACE TOKEN>'
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
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
    user_id = row['user_id']
    asin = row['asin']
    rating = row['rating']
    title = row['title']
    description = row['description']
    features = row['features']
    review = row['text']
    
                    
    #FOR BOOK MOVIES DOMAIN
    messages = [
    {"role": "system", "content": "You are an advanced AI model that extracts detailed, objective aspects of an item based on its title, description, features, and review. Your task is to identify and list these aspects, making careful inferences based on the provided data where appropriate. However, you must strictly avoid including any user sentiments, opinions, or personalized content. Focus solely on the requested aspects, ensuring that any inferences are based only on clear and explicit cues in the provided information. Do not stretch inferences or assumptions where details are vague or missing."},
    
    {"role": "user", "content": f"""Title: {title}
Description: {description}
Features: {features}
Review: {review}

Focus on the following aspects, based on the explicit information provided and **only make inferences when they are strongly supported by clear patterns in the data**:

1. Genre: Identify the genre of the item based on explicit information and clear inferences. If not mentioned or cannot be confidently inferred, state 'Not mentioned'.
2. Theme: Describe the main themes or topics discussed in the item, making logical inferences only if they are strongly supported by explicit cues in the data. If not mentioned or cannot be confidently inferred, state 'Not mentioned'.
3. Key Features: List any unique features or aspects of the item mentioned in the review or description, and make logical inferences about their significance only when clearly indicated. If not mentioned or cannot be inferred, state 'Not mentioned'.
4. Plot Summary: Provide a brief summary of the plot or storyline based on the description or review. Only make inferences where the storyline is clear, and avoid guessing. If not mentioned or cannot be inferred, state 'Not mentioned'.
5. Character Development: Describe the development and depth of the main characters as mentioned in the review or description, including any logical inferences only if strongly supported by clear evidence. If not mentioned or cannot be inferred, state 'Not mentioned'.
6. Setting and World-Building: Detail the setting and the world in which the story takes place, based on explicit details and clear inferences. If not mentioned or cannot be inferred, state 'Not mentioned'.
7. Target Audience: Identify the intended target audience for the item, making logical inferences based on explicit cues in the provided information. If not mentioned or cannot be inferred, state 'Not mentioned'.
8. Pacing and Style: Describe the pacing of the story and the writing or cinematic style based on explicit information and any logical inferences, only if strongly supported by clear details. If not mentioned or cannot be inferred, state 'Not mentioned'.

Important:Ensure the extracted aspects capture sufficient detail to analyze user behavior consistency across domains, focusing on comparable features like emotional tone, thematic complexity, and narrative depth.


Important: Exclude user sentiments or opinions and provide only the relevant aspects. If certain information cannot be inferred confidently, state 'Not mentioned' instead of making assumptions."""}
]

    
    '''     
    # FOR BOOK FOOD DOMAIN
    messages = [
    {"role": "system", "content": "You are an advanced AI model specialized in extracting detailed, objective aspects of items in both the Book and Food domains. Your task is to identify and list relevant aspects of each item based on its title, description, features, and review. Since Books and Food serve different functions, focus on identifying meaningful commonalities in experiential, emotional, and sensory engagement that apply across both domains. Avoid making abstract or irrelevant inferences."},
    
    {"role": "user", "content": f"""Title: {title}
Description: {description}
Features: {features}
Review: {review}

Please focus on the following aspects, ensuring that the information extracted is detailed, specific, and well-suited for finding relevant cross-domain connections between Books and Food. If any aspect is not mentioned or cannot be confidently inferred, explicitly state 'Not mentioned'.

1. **Theme/Cultural Significance**: Identify the main theme or cultural significance of the item. For Books, this might include the central message, cultural context, or broader significance. For Food, consider its role in cultural traditions, health implications, or its importance in heritage. Focus on finding shared values or themes like tradition, identity, or emotional impact across both domains. If not mentioned or cannot be confidently inferred, state 'Not mentioned'.

2. **Target Audience**: Identify the intended target audience for the item. For Books, focus on the readers' demographics or interests. For Food, consider consumers' health concerns, dietary preferences, or cultural background. Look for common values or lifestyles that align across both domains, such as authenticity, sustainability, or community-building. If not mentioned or cannot be confidently inferred, state 'Not mentioned'.

3. **Experience/Sensory Elements**: Describe the sensory or experiential aspects of the item. For Books, this could involve the emotional engagement or atmosphere created by the narrative, highlighting the emotional response it provokes in the reader. For Food, focus on taste, texture, and the overall sensory experience, particularly how these elements evoke emotional satisfaction or create lasting memories. Explore common ground in how both domains offer immersive, multi-sensory experiences that contribute to overall satisfaction and enjoyment. If not mentioned or cannot be confidently inferred, state 'Not mentioned'.

4. **Presentation/Storytelling**: Provide a detailed summary of how the item is presented. For Books, describe the narrative style, format, or visual elements (e.g., cover design). For Food, describe the packaging, branding, or any storytelling elements used in marketing. Draw comparisons based on how both domains use presentation to reflect values, heritage, or identity, fostering a deeper connection with the consumer. If not mentioned or cannot be confidently inferred, state 'Not mentioned'.

5. **Historical/Contextual Elements**: Detail the historical or contextual background of the item. For Books, this might include the historical period or cultural setting of the story. For Food, consider its origin, traditional preparation methods, or its role in cultural history. Focus on how both domains balance tradition with innovation, reflecting cultural evolution over time, and how this balance contributes to their relevance in modern culture. If not mentioned or cannot be confidently inferred, state 'Not mentioned'.

**Important**: Ensure that all comparisons between the Book and Food domains are grounded in shared themes, values, or experiences. Avoid forcing comparisons between unrelated elements. The goal is to identify meaningful commonalities that can apply across both domains, while maintaining objectivity and avoiding bias. If certain information is not mentioned or cannot be confidently inferred, state 'Not mentioned' rather than making assumptions.
"""}
]
'''

    
    '''    
    #FOR MOVIE MUSIC DOMAIN
    messages = [
    {"role": "system", "content": "You are an advanced AI model that extracts detailed, objective aspects of items in either the Movie or Music domain. Your task is to identify and list relevant aspects based on the data provided for each item. Use logical inferences to answer where possible. If there is no sufficient data to make an inference, only then mark it as 'Not mentioned'."},

    {"role": "user", "content": f"""Title: {title}
Description: {description}
Features: {features}
Review: {review}

Please focus on the following aspects based on the provided data. If information can be logically inferred from the data, provide the answer. If no data is available or no inference can be made, state 'Not mentioned'. **Do not mark an aspect as 'inferred' — just provide the answer or 'Not mentioned'**.

1. **Theme/Emotional Tone**: Identify the theme(s) and emotional tone based on the provided description and review. If not mentioned or cannot be confidently inferred, state 'Not mentioned'.
2. **Cultural/Contextual Significance**: Describe the cultural or contextual significance of the item based on the explicit information or reasonable inference from the data. If not mentioned or cannot be confidently inferred, state 'Not mentioned'.
3. **Target Audience and Appeal**: Identify the intended target audience and the item’s appeal. If not mentioned or cannot be confidently inferred, state 'Not mentioned'.
4. **Atmosphere and Aesthetic Style**: Describe the atmosphere and aesthetic style based on the explicit information or logical inferences from the data. If not mentioned or cannot be confidently inferred, state 'Not mentioned'.
5. **Narrative/Storytelling Elements**: Describe the narrative or storytelling elements based on the available information. If not mentioned or cannot be confidently inferred, state 'Not mentioned'.
6. **Rhythm and Pacing**: Describe the rhythm and pacing of the item. For Movies, this includes scene pacing; for Music, this refers to the tempo and flow. If not mentioned or cannot be confidently inferred, state 'Not mentioned'.
7. **Tradition vs. Innovation**: Identify whether the item reflects traditional elements or embraces innovation in its genre, style, or approach. Consider classic vs. modern techniques in both Movies and Music. If not mentioned or cannot be confidently inferred, state 'Not mentioned'.

**Important**: When making inferences, look for supporting patterns across the title, description, features, and review. Make an inference only when multiple aspects consistently suggest the same information. If there is no clear pattern or only weak signals, provide 'Not mentioned' to avoid forced or speculative conclusions. Incomplete or missing data may lead to incorrect embeddings and item comparisons. Exclude all user sentiments, opinions, or commentary.

"""}
]
    
    '''
    '''     
    #FOR ELECTRONICS FOOD DOMAIN
    messages = [
    {"role": "system", "content": "You are an advanced AI model specialized in extracting detailed, objective aspects of items in both the Electronics and Food domains. Your task is to identify and list relevant aspects of each item based on its title, description, features, and review. The focus is on extracting information that is comprehensive and relevant, with the goal of finding qualities that are comparable across these two specific domains. Draw comparisons based on broader themes, ideas, and nuances without referencing specific items. Avoid including any user sentiments, opinions, or personalized content. Concentrate solely on the requested aspects and do not provide any additional information."},
    {"role": "user", "content": f"""Title: {title}
Description: {description}
Features: {features}
Review: {review}

Please focus on the following aspects, ensuring that the information extracted is detailed, specific, and well-suited for cross-domain comparison between Electronics and Food. If any aspect is not mentioned or cannot be confidently inferred, state 'Not mentioned'.

1. **Functionality and Usage**: Describe the primary function and usage context of the item. For Electronics, this may include the main features and purposes of the device. For Food, consider how it is consumed, its nutritional benefits, or its role in a meal. Highlight the practical applications and how each item serves a specific purpose, adding value to the user’s experience. If not mentioned or cannot be confidently inferred, state 'Not mentioned'.

2. **Target Consumer Demographics**: Identify the target consumer group for the item. For Electronics, this could involve age groups, professional users, or lifestyle preferences. For Food, consider health concerns, dietary preferences, or cultural background. Discuss shared values such as convenience, health, or technological advancement, and how both domains cater to evolving consumer needs and preferences. If not mentioned or cannot be confidently inferred, state 'Not mentioned'.

3. **Experience and Sensory Interaction**: Describe the sensory or experiential elements associated with the item. For Electronics, this may include user interface, tactile feedback, or overall user experience. For Food, focus on taste, texture, and the sensory satisfaction it provides. Emphasize how both domains engage the consumer's senses and how these experiences contribute to satisfaction and enjoyment. If not mentioned or cannot be confidently inferred, state 'Not mentioned'.

4. **Design and Presentation**: Provide details on the design or presentation of the item. For Electronics, describe the aesthetic design, materials, or user interface layout. For Food, consider packaging, branding, or visual appeal. Explore how each domain uses design and presentation to create an appealing, user-friendly experience, reflecting brand values and consumer trends. If not mentioned or cannot be confidently inferred, state 'Not mentioned'.

5. **Technological or Production Background**: Detail the technological or production background of the item. For Electronics, this might include the technology used, manufacturing process, or environmental impact. For Food, consider preparation methods, sourcing, or sustainability aspects. Highlight how each domain incorporates technological innovation or ethical considerations into the creation of their products. If not mentioned or cannot be confidently inferred, state 'Not mentioned'.

6. **Tradition vs. Innovation**: Identify whether the item reflects traditional elements or embraces innovation in its genre, style, or approach. For Electronics, this could be modern technological advancements or classic designs. For Food, this could relate to traditional preparation methods or innovative new recipes. If not mentioned or cannot be confidently inferred, state 'Not mentioned'.

**Important**: Ensure that all comparisons between the Electronics and Food domains are based on broader themes, ideas, and nuances, without referencing specific items. Avoid including any user sentiments, opinions, or additional commentary. The goal is to extract comprehensive information that highlights meaningful connections across these two domains while maintaining objectivity and avoiding bias.
"""}
]
    
    '''
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    generated_text = generate_response(input_ids, model, tokenizer, min_length=50)

    if generated_text and is_response_valid(generated_text, min_length=50):
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

