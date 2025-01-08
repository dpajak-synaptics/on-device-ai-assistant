import subprocess
import json
from huggingface_hub import hf_hub_download
import os

# Disable tokenization parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Embeddings:
    def __init__(self):
        """
        Initializes the EmbeddingGenerator class by downloading the quantized model.
        """
        self.model_path = None
        self.demo_dir = os.path.dirname(__file__)
        self.models_dir = os.path.join(self.demo_dir, '../models/')
        self._download_model()

    def _download_model(self):
        """
        Downloads the quantized GGUF model from Hugging Face and stores the path.
        """
        try:
            # Ensure the models directory exists
            os.makedirs(self.models_dir, exist_ok=True)
            
            # Download the model and store its path
            self.model_path = hf_hub_download(
                repo_id="second-state/All-MiniLM-L6-v2-Embedding-GGUF",
                filename="all-MiniLM-L6-v2-Q8_0.gguf",
                local_dir=self.models_dir
            )
            print(f"Model downloaded to: {self.model_path}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            exit(1)

    def generate(self, text):
        """
        Generates embeddings for the given text using the quantized model.

        Args:
            text (str): The input text for which to generate embeddings.

        Returns:
            list: A list representing the generated embedding.
        """
        try:
            # Construct the full command
            llama_bin_path = os.path.join(self.demo_dir, '../models/llama.cpp/build/bin/llama-embedding')
            command = [
                llama_bin_path,
                '-m', self.model_path,
                '-p', text,
                '--ctx-size', '128',
                '--batch-size', '512',
                '--embd-output-format', 'json',
                '-t', '4'
            ]

            # Run llama-embedding command
            result = subprocess.run(command, capture_output=True, text=True)

            if result.returncode != 0:
                print("Error: Llama-embedding command failed to execute successfully.")
                print("Return code:", result.returncode)
                print("Error output:", result.stderr)
                raise RuntimeError(f"Llama-embedding command failed\n{command}")

            if not result.stdout:
                print("Error: No output from llama-embedding command.")
                raise ValueError("No output from llama-embedding")

            # Parse the JSON output and return the embedding
            embedding_data = json.loads(result.stdout)
            embedding = embedding_data.get('data', [{}])[0].get('embedding')

            if embedding is None:
                print("Error: Embedding data is missing in the output.")
                raise ValueError("No embedding data found")

            return embedding

        except Exception as e:
            print(f"Error generating embedding: {e}")
            exit(1)

# Example usage
if __name__ == "__main__":
    generator = Embeddings()
    sample_text = "This is a sample text for generating embeddings."
    embedding = generator.generate(sample_text)
    print("Generated Embedding:", embedding)
