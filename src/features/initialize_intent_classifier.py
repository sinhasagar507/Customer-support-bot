# Importing the necessary libraries
import re 
import contractions as cm 
from nltk.tokenize import TweetTokenizer #type: ignore
import pickle as pkl 
import torch 
from ..models.define_model import IntentClassifier


# Create a blank Tokenizer with just the English vocab
tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

def clean_text(
        text, words=True, stops=True, urls=True, tags=True, hashtags = True, punctuations=True,  
        newLine=True, ellipsis=True, special_chars=True, condensed=True, non_breaking_space=True, 
        character_encodings=True, stopwords=True, only_words=True) -> str:
    
    """ Clean tweets after extracting all hashtags and username tags
    Not comprehensive enough to capture all idiosyncrasies, but works for most of the time
    """
    
    # Capture only words and no numbers
    if words:
        pattern = r"\d"
        text = re.sub(pattern, "", text)
        
    # Remove URLs 
    if urls:
        pattern = "(https\:)*\/*\/*(www\.)?(\w+)(\.\w+)\/*\w*"
        text = re.sub(pattern, "", text)
        
    # Remove tags 
    if tags:
        text = re.sub("@\S+", "", text)
        
    # Remove hashtags 
    if hashtags: 
        text = re.sub("#\w+", "", text)
        
    # Remove punctuations
    if punctuations:
        for punct in puncts: 
            text = text.replace(punct, "")
        
    # Replacing one or more occurrences of '\n' with ''
    # Replacing multiple occurrences, i.e., >=2 occurrences with '.'
    if newLine:
        text = re.sub("\n+", "", text)
        text = re.sub(r'\.\s+', '.', text)
        
    # Fix contractions
    if condensed:
        try:
            text = cm.fix(text)
        except: 
            print(text)
        
    # Remove non-breaking space 
    if non_breaking_space: 
        pattern = r"(\xa0|&nbsp)"
        text = re.sub(pattern, "", text)
        
    # Remove stopwords
    # if stopwords:
    #     text = text.lower()
    #     # print(f"Original Shape of the Data is {.shape}")
        
    #     # Splitting with NLTK's Tweet tokenizer. This limits repeated characters to 
    #     # three with the reduce lens parameter and strips all the "@'s". It also splits 
    #     # it into 1-gram tokens         
    #     words = tokenizer.tokenize(text)
    #     filtered_words = [word for word in words if word not in eng_stopwords]
    #     text = " ".join(words)
    #     text = text.strip()  # Add further checks for cleaning 
    
    return text

# List of manual stopwords
manual_stopwords = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
}
    
def basic_preprocess_tokens(tokens): 
    # Convert to lowercase
        
    # Convert string representation of list to actual list
    # tokens = ast.literal_eval(tokens)
    
    # Convert to lowercase
    tokens = [token.lower() for token in tokens]
    
    # Remove punctuation
    tokens = [token.translate(str.maketrans('', '', string.punctuation)) for token in tokens]
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in manual_stopwords]
    
    return tokens

def load_model(): 
    # Load the embedding matrix
    with open("../../objects/embedding_matrix.pkl", "rb") as file:
        embedding_matrix = pkl.load(file)
        
    
    # Initialize and load the model 
    model = IntentClassifier(embedding_matrix)
    checkpoint = torch.load("../../models/intent_classification_model.pt")  
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model

    def inference(tokens): 
        

# Load the vocabulary and label encoder
# Load the trained model 
def inference(tokens):
    """ 
    Perform preprocessing and inference on the input text using the trained model.
    
    Parameters:
    - model: The trained PyTorch model for intent classification.
    - text: The input text string.
    - vocabulary: A dictionary mapping tokens to indices.
    - seq_len: The fixed sequence length expected by the model.
    
    Returns:
    - pred_label: The predicted label index.
    """

    # Preprocess the text
    # tokens = text.split()
    indices = [vocabulary.get(token, vocabulary["<unknown>"]) for token in tokens]  # Use 0 for unknown words
    padded_indices = indices[:seq_len] + [0] * max(0, seq_len - len(indices))  # Pad with zeros
    print(padded_indices)
    input_tensor = torch.tensor(padded_indices).unsqueeze(0)  # Add batch dimension
    print(input_tensor)
    # print(input_tensor.shape)
    
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        label = torch.argmax(output, 1)
        label_text = label_encoder.inverse_transform(label)
    
    return label_text

if __name__=="main":
    
    # Write down all the stopwords and punctuation marks for removal 
    manual_stopwords = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
}
    
    # Punctuation marks to remove
    puncts = ['\u200d', '?', '....','..','...','','@','#', ',', '.', '"', ':', ')', '(', '-', '!', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '*', '+', '\\', 
        '•', '~', '£', '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', 
        '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', 
        '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 
        'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', 
        '¹', '≤', '‡', '√', '!','🅰','🅱']
    
    
    
    



