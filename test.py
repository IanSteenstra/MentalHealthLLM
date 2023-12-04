import os
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain.chat_models import ChatOpenAI

# Download the necessary NLTK models and data
nltk.download('punkt')
nltk.download('stopwords')

# Retrieve the set of English stopwords
stop_words = set(stopwords.words('english'))

# Initialize the LLM (here we assume GPT-3.5 or compatible API)
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Folder where the corpus is stored
CORPUS_FOLDER = "corpus"

MAX_RESPONSE_SIZE = 200

# Tokenize documents, remove stopwords and non-alphabetic tokens using NLTK
def tokenize(document):
    # Tokenize the document
    tokens = word_tokenize(document.lower())
    # Remove stopwords and non-alphabetic tokens
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return filtered_tokens

# Load and tokenize documents from the "corpus" folder
def load_documents():
    print("Loading and tokenizing documents...")
    tokenized_documents = []
    for filename in os.listdir(CORPUS_FOLDER):
        if filename.endswith('.txt'):  # Ensure we're reading text files
            with open(os.path.join(CORPUS_FOLDER, filename), 'r', encoding='iso-8859-1') as file:
                # Read and tokenize the document
                text = file.read()
                tokens = tokenize(text)
                tokenized_documents.append(tokens)
    print(f"Loaded and tokenized {len(tokenized_documents)} documents.")
    return tokenized_documents

# BM25 retrieval function
def retrieve_documents_with_bm25(query, tokenized_documents):
    print("Retrieving documents with BM25...")
    bm25 = BM25Okapi(tokenized_documents)
    query_tokens = tokenize(query)
    doc_scores = bm25.get_scores(query_tokens)
    top_doc_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:3] #TODO: Change to k as a param
    top_documents = [tokenized_documents[i] for i in top_doc_indices]
    print("Documents retrieved.")
    return top_documents

#TODO: Add batching
# Function to analyze document chunks with LLM independently
def analyze_document_chunks(document, user_query, max_token_limit=4096):
    # Prepare the initial part of the prompt
    prompt_intro = f"The user is seeking information related to '{user_query}'. Given this context, carefully analyze the following text. Identify and summarize the key points that directly address the user's query in less than 100 words, especially focusing on any relevant facts, insights, or advice. Highlight critical details and provide a concise summary that would be most helpful and informative to the user:\n\n"
    
    # Tokenize the document
    document_tokens = nltk.word_tokenize(document)
    
    # Estimate the number of tokens in the prompt
    estimated_prompt_tokens = len(nltk.word_tokenize(prompt_intro))

    # Calculate the maximum size for each chunk in terms of tokens
    max_chunk_token_size = max_token_limit // 2 - estimated_prompt_tokens - MAX_RESPONSE_SIZE #TODO - Fix

    # Create chunks of tokens
    token_chunks = [document_tokens[i:i + max_chunk_token_size] for i in range(0, len(document_tokens), max_chunk_token_size)]
    analysis_results = []

    print(f"Analyzing {len(token_chunks)} document chunks...")
    for chunk in token_chunks:
        # Reassemble the tokens into a string
        chunk_text = ' '.join(chunk)
        prompt = prompt_intro + chunk_text

        # Generate analysis for each chunk
        analysis = llm.predict(prompt, max_tokens=MAX_RESPONSE_SIZE//2) #TODO - Fix

        print(f"Chunk Completed! Token Size({len(nltk.word_tokenize(prompt)) +  MAX_RESPONSE_SIZE})") #TODO - Fix
        analysis_results.append(analysis)

    print("Document chunks analyzed.")
    return analysis_results

# Main function to run the retrieval system
def run_system():
    user_query = input("Please enter your mental health-related query: ")
    tokenized_documents = load_documents()

    # Retrieve top k documents using BM25
    top_documents = retrieve_documents_with_bm25(user_query, tokenized_documents)[:2]  # Change here for top k documents

    all_chunk_responses = []
    # Analyze each top document by chunks independently
    for tokenized_doc in top_documents:
        # Convert tokenized document back to string
        doc_string = ' '.join(tokenized_doc)
        
        #TODO: Different preprocessing for doc_string for model vs BM25 algo
        all_chunk_responses += analyze_document_chunks(doc_string, user_query)

    # Combine the independent chunk analyses
    combined_analysis = "\n\n---\n\n".join(all_chunk_responses)

    final_prompt = (
        f"Summary of Information: {combined_analysis}\n\n"
        "Based on this earlier summary about cognitive-behavioral therapy (CBT) techniques, "
        f"apply these strategies directly through a normal conversational style of a counselor in 100 words or less to help a user's current situation and questions: {user_query}"
    )

    final_response = llm.predict(final_prompt, max_tokens=MAX_RESPONSE_SIZE)

    print(final_response)


if __name__ == "__main__":
    run_system()