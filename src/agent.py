import json
import os
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from src.generate_embedding import Embeddings
from src.utils import file_checksum
from collections import Counter
import re

DEMO_DIR = os.path.dirname(__file__)
BASE_QA = f'{DEMO_DIR}/../data/base.json'

MARS_DATA = f'{DEMO_DIR}/../data/mars.json'
DISHWASHER_DATA = f'{DEMO_DIR}/../data/dishwasher.json'

ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"

class Agent:
    def __init__(self, threshold=0.39, keyword_weight=0.5, semantic_weight=1.2, qa_file=DISHWASHER_DATA):
        self.threshold = threshold
        self.embeddings = Embeddings()
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.qa_file = qa_file
        self.stop_words = {
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while", 
            "of", "to", "in", "on", "at", "by", "for", "with", "about", "as", "into", 
            "like", "through", "after", "over", "between", "out", "against", "during", 
            "without", "within", "under", "above", "up", "down", "off", "near", "this", 
            "that", "these", "those", "is", "am", "are", "was", "were", "be", "been", 
            "being", "have", "has", "had", "do", "does", "did", "can", "could", "shall", 
            "should", "will", "would", "may", "might", "must", "ought", "i", "you", "he", 
            "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", 
            "his", "her", "its", "our", "their", "mine", "yours", "hers", "ours", 
            "theirs", "what", "which", "who", "whom", "this", "that", "these", "those", 
            "there", "here", "when", "where", "why", "how", "all", "any", "both", 
            "each", "few", "many", "more", "most", "some", "such", "no", "nor", "not", 
            "only", "own", "same", "so", "than", "too", "very", "computer"
        }
        self.qa_pairs = self.load_manual_data()
        self.question_embeddings = self.load_embeddings(self.qa_pairs)

    def change_agent(self, qa_file):
        self.qa_file = qa_file
        self.qa_pairs = self.load_manual_data()
        self.question_embeddings = self.load_embeddings(self.qa_pairs)

    def preprocess_text(self, text):
        """Preprocess text by tokenizing, normalizing, and removing stop words."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\\s]', '', text)  # Remove non-alphanumeric characters
        tokens = text.split()
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        return filtered_tokens

    def keyword_match_score(self, query, question_keywords):
        """Calculate a simple keyword overlap score."""
        query_tokens = Counter(self.preprocess_text(query))
        intersection = sum((query_tokens & question_keywords).values())
        total = sum(query_tokens.values())
        return intersection / total if total > 0 else 0
    
    def load_manual_data(self):
        """Load QA data from a JSON file, cache it as a TSV file, and extract keywords."""

        with open(BASE_QA, 'r') as file:
            base = json.load(file)

        with open(self.qa_file, 'r') as file:
            file = json.load(file)

        manual_data = base['qa_pairs'] + file['qa_pairs']
        self.valQuestions = file['validation']['questions']
        self.valAnswers = file['validation']['answers']
        self.voiceModel = file['voice']['onnx_file']
        self.voiceJson = file['voice']['json_file']
        self.threshold = file['threshold']

        # Pre-extract keywords for all questions
        for pair in manual_data:
            pair['keywords'] = Counter(self.preprocess_text(pair['question']))

        qa_df = pd.DataFrame(manual_data)
        qa_df.to_csv(f'{DEMO_DIR}/../cached/metadata.tsv', index=False, header=True, sep='\t')

        return manual_data

    def load_embeddings(self, qa_pairs):
        """Load or generate embeddings for QA pairs."""
        filename = f'{DEMO_DIR}/../cached/vectors-{file_checksum(json.dumps(qa_pairs, sort_keys=True))}.tsv'

        if os.path.exists(filename):
            embeddings = pd.read_csv(filename, sep='\t', header=None).values
            return embeddings

        print("Generating question embeddings...")
        questions = [pair["question"] + " " + pair["answer"] for pair in qa_pairs]
        question_embeddings = np.array([self.embeddings.generate(q) for q in tqdm(questions)])

        embedding_df = pd.DataFrame(question_embeddings)
        embedding_df.to_csv(filename, sep='\t', index=False, header=False)

        return question_embeddings

    def keyword_match_score(self, query, question_keywords):
        """Calculate a simple keyword overlap score."""
        query_tokens = Counter(self.preprocess_text(query))
        intersection = sum((query_tokens & question_keywords).values())
        total = sum(query_tokens.values())
        return intersection / total if total > 0 else 0

    def answer_query(self, query):
        """Find the best answer to a query using combined semantic and keyword search."""

        start_embedding = time.time()
        query_embedding = self.embeddings.generate(query)
        end_embedding = time.time()
        print(f"{ANSI_YELLOW}Embedding Time: {(end_embedding - start_embedding) * 1000:.2f} ms{ANSI_RESET}")


        if query_embedding is None:
            return "Sorry, I couldn't generate an embedding for your query."

        # Semantic similarity
        similarities = cosine_similarity([query_embedding], self.question_embeddings).flatten()

        # Keyword similarity
        keyword_scores = [
            self.keyword_match_score(query, pair["keywords"])
            for pair in self.qa_pairs
        ]

        # Combined score
        combined_scores = (
            self.semantic_weight * similarities + self.keyword_weight * np.array(keyword_scores)
        )

        best_match_idx = np.argmax(combined_scores)
        best_match_answer = self.qa_pairs[best_match_idx]["answer"]
        best_match_similarity = float(combined_scores[best_match_idx])

        return {
            'answer': best_match_answer,
            'similarity': best_match_similarity,
            'semantic_similarity': float(similarities[best_match_idx]),
            'keyword_score': keyword_scores[best_match_idx]
        }

    def run_command_tokens(self, answer):
        """Process and replace command tokens in the answer."""

        # Check for specific token to change agent
        if "{load_mars}" in answer:
            self.change_agent(MARS_DATA)
            time.sleep(0.2)
            return "Mars mission ships computer demo active."

        if "{load_dishwasher}" in answer:
            self.change_agent(DISHWASHER_DATA)
            time.sleep(0.2)
            return "Dishwasher AI assistant demo active."

        tools_file = f'{DEMO_DIR}/../data/tools.json'
        if not os.path.exists(tools_file):
            return answer

        with open(tools_file) as f:
            token_command_list = json.load(f)

        for item in token_command_list:
            token = item.get("token")
            command = item.get("command")
            if token and command and token in answer:
                response = os.popen(command).read().strip()
                if response:
                    answer = answer.replace(token, response)

        return answer

    def handle_query(self, query):
        """Process text query and generate a response."""
        result = self.answer_query(query)
        answer = result['answer']
        similarity = result['similarity']

        # Process commands in answer if threshold is met
        if similarity >= self.threshold:
            answer = self.run_command_tokens(answer)

        return {
            "answer": answer,
            "confidence": similarity
        }