import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from src.generate_embedding import generate_embedding
from src.utils import file_checksum

DEMO_DIR = os.path.dirname(__file__)
QA_DATA = f'{DEMO_DIR}/../data/question_answers.json'

class Agent:
    def __init__(self):
        self.qa_pairs = self.load_manual_data()
        self.question_embeddings = self.load_embeddings(self.qa_pairs)

    def load_manual_data(self):
        """Load QA data from a JSON file and cache it as a TSV file."""
        filename = QA_DATA
        with open(filename, 'r') as file:
            manual_data = json.load(file)

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
        questions = [pair["question"] for pair in qa_pairs]
        question_embeddings = np.array([generate_embedding(q) for q in tqdm(questions)])

        embedding_df = pd.DataFrame(question_embeddings)
        embedding_df.to_csv(filename, sep='\t', index=False, header=False)

        return question_embeddings

    def answer_query(self, query):
        """Find the best answer to a query using cosine similarity."""
        query_embedding = generate_embedding(query)

        if query_embedding is None:
            return "Sorry, I couldn't generate an embedding for your query."

        similarities = cosine_similarity([query_embedding], self.question_embeddings).flatten()
        threshold = max(0.3, similarities.mean() - similarities.std())

        if similarities.max() < threshold:
            return "Sorry, I don't know the answer to that question."

        best_match_idx = np.argmax(similarities)
        return self.qa_pairs[best_match_idx]["answer"]

    def run_command_tokens(self, answer):
        """Process and replace command tokens in the answer."""
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
        # Generate raw answer
        raw_answer = self.answer_query(query)

        # Process commands in answer
        processed_answer = self.run_command_tokens(raw_answer)

        return processed_answer