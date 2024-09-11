import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine import *

# Assuming all your previous functions (fetch_arxiv_data, hybrid_recommendations, evaluate_model, etc.) are defined and imported here.

# Function to display recommended papers in tabs
def display_recommendations(ranked_papers):
    if ranked_papers:
        for i, (paper, score) in enumerate(ranked_papers, 1):
            with st.expander(f"Paper {i}: {paper['title']} (Score: {score:.2f})"):
                st.markdown(f"**Title**: {paper['title']}")
                st.markdown(f"**Authors**: {', '.join(paper['author'])}")
                st.markdown(f"**Link**: [Read Here]({paper['link']})")
                st.markdown(f"**Published Date**: {paper['published_date']}")
                st.markdown("---")

# Function to display analytics (graphical visualization and evaluation metrics)
def display_analytics(paper_titles, content_scores, collaborative_scores, combined_scores, true_labels, predicted_labels):
    # Visualize recommendation scores
    y_pos = np.arange(len(paper_titles))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(y_pos, content_scores, align='center', alpha=0.5, label='Content-Based')
    ax.barh(y_pos, collaborative_scores, align='center', alpha=0.5, label='Collaborative Filtering')
    ax.barh(y_pos, combined_scores, align='center', alpha=0.5, label='Hybrid')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(paper_titles)
    ax.invert_yaxis()  # Invert Y-axis to match ranking order
    ax.set_xlabel('Recommendation Scores')
    ax.set_title('Comparison of Recommendation Scores')
    ax.legend()

    st.pyplot(fig)

    # Display evaluation metrics (precision, recall, F1-score)
    st.markdown("### Evaluation Metrics")
    precision, recall, f1 = evaluate_model(true_labels, predicted_labels)
    st.write(f"**Precision**: {precision:.2f}")
    st.write(f"**Recall**: {recall:.2f}")
    st.write(f"**F1 Score**: {f1:.2f}")

# Main Streamlit App
def main():
    st.set_page_config(page_title="Scientific Paper Recommendation System", layout="wide")
    st.title("Hybrid Recommendation System for Scientific Papers")

    # Sidebar Navigation
    pages = ["Home/Recommendations", "Analytics"]
    page = st.sidebar.selectbox("Navigate", pages)

    # User input fields for the Home/Recommendations page
    if page == "Home/Recommendations":
        st.header("Get Paper Recommendations")
        search_query = st.text_input("Enter Search Query", placeholder="e.g., Machine Learning")
        user_id = st.number_input("Enter User ID (for collaborative filtering)", min_value=0, value=0)
        max_results = st.slider("Number of Results to Fetch", min_value=1, max_value=20, value=5)

        if st.button("Get Recommendations"):
            if search_query:
                # Fetch papers and get recommendations
                papers = fetch_arxiv_data(search_query, max_results)  # You can include other APIs here
                if papers:
                    ranked_papers, content_scores, collaborative_scores, combined_scores = hybrid_recommendations(user_id, search_query, papers, interaction_matrix)
                    st.subheader(f"Top {max_results} Recommended Papers")
                    display_recommendations(ranked_papers[:max_results])  # Display the top N rec
                else:
                    st.warning("No papers found for the given query.")
            else:
                st.warning("Please enter a search query.")

    # Analytics page
    elif page == "Analytics":
        st.header("Analytics: Graphical Visualizations and Evaluation")
        
        # Example placeholder data for visualization
        paper_titles = ["Paper 1", "Paper 2", "Paper 3", "Paper 4", "Paper 5"]  # Use real paper titles
        content_scores = [0.6, 0.7, 0.4, 0.8, 0.5]  # Placeholder scores from content-based filtering
        collaborative_scores = [0.5, 0.6, 0.3, 0.7, 0.4]  # Placeholder collaborative scores
        combined_scores = [0.55, 0.65, 0.35, 0.75, 0.45]  # Placeholder combined (hybrid) scores
        true_labels = [1, 1, 0, 1, 0]  # Placeholder for actual user feedback
        predicted_labels = [1, 1, 0, 1, 0]  # Placeholder for predictions

        display_analytics(paper_titles, content_scores, collaborative_scores, combined_scores, true_labels, predicted_labels)

if __name__ == "__main__":
    main()
