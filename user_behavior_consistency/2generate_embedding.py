import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import argparse
from sklearn.preprocessing import normalize

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    """Function to generate BERT embedding using the [CLS] token for a given text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    bert_model.eval()

    with torch.no_grad():
        outputs = bert_model(**inputs)
        last_hidden_states = outputs.last_hidden_state

    # Use the [CLS] token embedding (first token)
    cls_embedding = last_hidden_states[:, 0, :].squeeze(0).detach().numpy()

    return cls_embedding

def get_structure_embedding():
    """Function to generate the BERT embedding for the common structure."""
        
    #FOR BOOK MOVIE DOMAIN
    common_structure_text = """
    1. Genre: Not mentioned
    2. Theme: Not mentioned
    3. Key Features: Not mentioned
    4. Plot Summary: Not mentioned
    5. Character Development: Not mentioned
    6. Setting and World-Building: Not mentioned
    7. Target Audience: Not mentioned
    8. Pacing and Style: Not mentioned
    """

    ''' 
    #FOR BOOK FOOD DOMAIN
    common_structure_text = """
    1. Theme/Cultural Significance: Not mentioned
    2. Target Audience: Not mentioned
    3. Experience/Sensory Elements: Not mentioned
    4. Presentation/Storytelling: Not mentioned
    5. Historical/Contextual Elements: Not mentioned
    """
    '''

    '''    
    #FOR MOVIE MUSIC DOMAIN
    common_structure_text = """
    1. Theme/Emotional Tone: Not mentioned
    2. Cultural/Contextual Significance: Not mentioned
    3. Target Audience and Appeal: Not mentioned
    4. Atmosphere and Aesthetic Style: Not mentioned
    5. Narrative/Storytelling Elements: Not mentioned
    6. Rhythm and Pacing: Not mentioned
    7. Tradition vs. Innovation: Not mentioned
    """
    '''
      
    '''
    #FOR ELECTRONIC FOOD DOMAIN
    common_structure_text = """
    1. Functionality and Usage: Not mentioned
    2. Target Consumer Demographics: Not mentioned
    3. Experience and Sensory Interaction: Not mentioned
    4. Design and Presentation: Not mentioned
    5. Technological or Production Background: Not mentioned
    6. Tradition vs. Innovation: Not mentioned
    """
    '''

    return get_bert_embedding(common_structure_text)

def main(input_file, output_file, subtract_structure, normalize_embeddings):
    # Convert string inputs to boolean
    subtract_structure = subtract_structure.lower() == 'true'
    normalize_embeddings = normalize_embeddings.lower() == 'true'

    # Load the data
    df = pd.read_csv(input_file)

    # Optionally generate the structure embedding
    structure_embedding = get_structure_embedding() if subtract_structure else None

    # Normalize structure embedding if needed
    if subtract_structure and normalize_embeddings and structure_embedding is not None:
        structure_embedding = normalize(structure_embedding.reshape(1, -1))[0]

    # Generate embeddings for the 'Generated Response' column with a progress bar
    embeddings = []
    for response in tqdm(df['Generated_Response'], desc="Generating embeddings"):
        embedding = get_bert_embedding(response)

        # Normalize item embedding if needed
        if normalize_embeddings:
            embedding = normalize(embedding.reshape(1, -1))[0]

        # Subtract the structure embedding if needed
        if subtract_structure and structure_embedding is not None:
            embedding -= structure_embedding

        embeddings.append(embedding)

    # Add the embeddings as a new column
    df['embedding'] = embeddings

    # Select the specified columns
    result_df = df[['user_id', 'asin', 'rating', 'Title','Generated_Response', 'embedding']]

    # Save the result to a new CSV file
    result_df.to_csv(output_file, index=False)

    print(f"Embeddings have been successfully generated and saved to '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate BERT embeddings for a CSV file.')
    parser.add_argument('--input_file', type=str, help='The input CSV file name.')
    parser.add_argument('--output_file', type=str, help='The output CSV file name.')
    parser.add_argument('--subtract_structure', type=str, default='false', help='Set to "true" to subtract the structure embedding.')
    parser.add_argument('--normalize_embeddings', type=str, default='false', help='Set to "true" to normalize embeddings before subtraction.')

    args = parser.parse_args()
    main(args.input_file, args.output_file, args.subtract_structure, args.normalize_embeddings)

