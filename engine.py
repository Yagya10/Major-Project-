import requests
import xml.etree.ElementTree as ET
import re
import nltk
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# 1. Fetch papers from multiple APIs (arXiv, Semantic Scholar, CrossRef, IEEE Xplore)
def fetch_arxiv_data(search_query, max_results=5):
    base_url = 'http://export.arxiv.org/api/query?'
    query = f'search_query={search_query}&max_results={max_results}'
    response = requests.get(base_url + query)
    
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        papers = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text
            abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text
            author = [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
            published_date = entry.find('{http://www.w3.org/2005/Atom}published').text
            link = entry.find('{http://www.w3.org/2005/Atom}id').text
            papers.append({'title': title, 'abstract': abstract, 'author': author, 'link': link, 'published_date': published_date})
            if len(papers) >= max_results:
                break
        return papers
    else:
        print("Failed to fetch data from arXiv")
        return []

# Similar fetch functions for other APIs (Semantic Scholar, CrossRef, IEEE Xplore)...

# 2. Text cleaning and preprocessing functions
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\b\d+\b', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

def tokenize_text(text):
    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return tokens

# 3. TF-IDF Vectorization and Cosine Similarity (Content-Based Filtering)
def vectorize_text(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

# **4. Collaborative Filtering - Adding Implicit Feedback-Based Recommendations**

# Sample interaction matrix: User-Paper interactions (implicit feedback, e.g., views or clicks)
interaction_matrix = np.array([
    [1, 0, 3, 0, 5],
    [0, 2, 0, 0, 1],
    [4, 0, 0, 1, 0],
    [0, 5, 0, 0, 0],
    [2, 1, 0, 4, 0]
])  # Rows: Users, Columns: Papers

def collaborative_filtering(interaction_matrix):
    # Convert the interaction matrix to float type
    interaction_matrix = interaction_matrix.astype(float)
    # Perform Matrix Factorization using SVD (Singular Value Decomposition)
    U, sigma, Vt = svds(interaction_matrix, k=2)
    sigma = np.diag(sigma)
    
    # Reconstruct interaction matrix to predict user-item interactions
    predicted_matrix = np.dot(np.dot(U, sigma), Vt)
    return predicted_matrix

# Hybrid Recommendation (combine content-based + collaborative filtering)
def hybrid_recommendations(user_id, query, papers, interaction_matrix):
    # Clean and vectorize the abstracts for content-based filtering
    cleaned_abstracts = [clean_text(paper['abstract']) for paper in papers]
    tokenized_abstracts = [' '.join(tokenize_text(abstract)) for abstract in cleaned_abstracts]
    
    # Vectorize the tokenized abstracts (TF-IDF)
    tfidf_matrix, vectorizer = vectorize_text(tokenized_abstracts)
    
    # Cosine similarity between query and documents (Content-based)
    query_cleaned = clean_text(query)
    query_vectorized = vectorizer.transform([query_cleaned])
    content_similarities = cosine_similarity(query_vectorized, tfidf_matrix).flatten()
    
    # Collaborative Filtering Predictions for User-Paper Interactions
    collaborative_predictions = collaborative_filtering(interaction_matrix)
    user_predictions = collaborative_predictions[user_id]  # Get predictions for the given user

    # Ensure both content_similarities and user_predictions have the same length
    min_length = min(len(content_similarities), len(user_predictions))
    content_similarities = content_similarities[:min_length]  # Truncate to minimum length
    user_predictions = user_predictions[:min_length]  # Truncate to minimum length

    # Combine Content-Based and Collaborative Filtering Scores
    combined_scores = 0.5 * content_similarities + 0.5 * user_predictions
    
    # Sort papers by combined score
    ranked_papers = sorted(list(zip(papers[:min_length], combined_scores)), key=lambda x: x[1], reverse=True)
    
    return ranked_papers, content_similarities, user_predictions, combined_scores

# **5. Visualization of the recommendation scores (Graphical Display)**
def visualize_scores(paper_titles, content_scores, collaborative_scores, combined_scores):
    y_pos = np.arange(len(paper_titles))
    
    plt.figure(figsize=(12, 6))
    plt.barh(y_pos, content_scores, align='center', alpha=0.5, label='Content-Based')
    plt.barh(y_pos, collaborative_scores, align='center', alpha=0.5, label='Collaborative Filtering')
    plt.barh(y_pos, combined_scores, align='center', alpha=0.5, label='Hybrid')
    
    plt.yticks(y_pos, paper_titles)
    plt.xlabel('Recommendation Scores')
    plt.title('Comparison of Recommendation Scores')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()

# **6. Evaluation Metrics (Precision, Recall, F1-Score)**
def evaluate_model(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

# Main Program
if __name__ == "__main__":
    # Take search query and user ID as input
    search_query = input("Enter the search query: ")
    user_id = int(input("Enter user ID (for collaborative filtering): "))
    max_results = int(input("Enter the number of results to fetch from each source: "))
    
    # Fetch papers from multiple APIs
    papers = fetch_arxiv_data(search_query, max_results)  # Add other API calls similarly
    
    if papers:
        # Hybrid Recommendations (content-based + collaborative filtering)
        ranked_papers, content_scores, collaborative_scores, combined_scores = hybrid_recommendations(user_id, search_query, papers, interaction_matrix)
        
        # Display top ranked papers
        for i, (paper, score) in enumerate(ranked_papers[:10], 1):
            print(f"Rank {i}: {paper['title']} (Score: {score:.2f})")
            print(f"Authors: {', '.join(paper['author'])}")
            print(f"Link: {paper['link']}")
            print("-" * 80)
        
        # Extract paper titles for visualization
        paper_titles = [paper['title'] for paper, _ in ranked_papers[:10]]
        
        # Visualize recommendation scores
        visualize_scores(paper_titles, content_scores[:10], collaborative_scores[:10], combined_scores[:10])
        
        # **Evaluation example (mock true labels and predictions for demonstration)**
        true_labels = np.random.randint(2, size=10)  # Example of actual user engagement (0 or 1)
        predicted_labels = np.where(np.array(combined_scores[:10]) > 0.5, 1, 0)  # Mock predicted labels based on scores
        
        # Evaluate model performance
        evaluate_model(true_labels, predicted_labels)
        
    else:
        print("No papers found.")
