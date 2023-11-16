import os
from rank_bm25 import BM25Okapi
from langchain.llms import LLM
# Assume other necessary imports for tokenization, etc.

# Initialize the LLM (here we assume GPT-3.5 or compatible API)
llm = LLM('openai', api_key='YOUR_API_KEY')

# Folder where the documents are stored
INFO_FOLDER = "info"

# Load documents from the "info" folder
def load_documents():
    documents = []
    for filename in os.listdir(INFO_FOLDER):
        if filename.endswith('.txt'):  # Ensure we're reading text files
            with open(os.path.join(INFO_FOLDER, filename), 'r') as file:
                documents.append(file.read())
    return documents

# Tokenize documents - this is a placeholder function
def tokenize(document):
    # Placeholder for tokenization logic, e.g., using NLTK or spaCy
    return document.lower().split()

# BM25 retrieval function
def retrieve_documents_with_bm25(query, documents):
    tokenized_corpus = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = tokenize(query)
    doc_scores = bm25.get_scores(query_tokens)
    top_doc_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:3]
    top_documents = [documents[i] for i in top_doc_indices]
    return top_documents

# Function to analyze document chunks with LLM independently
def analyze_document_chunks(document, user_query, max_chunk_size):
    # Split the document into chunks that fit within the max_chunk_size
    chunks = [document[i:i+max_chunk_size] for i in range(0, len(document), max_chunk_size)]
    analysis_results = []
    for chunk in chunks:
        prompt = f"The user has asked: '{user_query}'. Analyze the following document chunk and gather information that is most relevant to the user's request:\n\n{chunk}"
        analysis = llm.generate(prompt, max_tokens=3000)  # We set max_tokens to 3000 as requested
        analysis_results.append(analysis)
    return analysis_results

# Main function to run the retrieval system
def run_system():
    user_query = input("Please enter your mental health-related query: ")
    documents = load_documents()

    # Retrieve top 3 documents using BM25
    top_documents = retrieve_documents_with_bm25(user_query, documents)

    all_chunk_responses = []
    # Analyze each top document by chunks independently
    for doc in top_documents:
        chunk_responses = analyze_document_chunks(doc, user_query, max_chunk_size=2900)  # Subtract some tokens for the prompt
        all_chunk_responses.extend(chunk_responses)

    # Combine the independent chunk analyses
    combined_analysis = "\n\n---\n\n".join(all_chunk_responses)

    # Use the combined analysis for the final output
    final_response = llm.generate(f"The following sections contain information relevant to your query about mental health:\n\n{combined_analysis}", max_tokens=3000)
    print(final_response)

if __name__ == "__main__":
    run_system()
