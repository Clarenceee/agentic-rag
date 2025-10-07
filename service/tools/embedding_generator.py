from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from process.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """A class to generate text embeddings using Sentence Transformers."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: Optional[str] = None):
        """Initialize the embedding generator with a Sentence Transformers model.

        Args:
            model_name: Name of the Sentence Transformers model to use.
                       Defaults to 'BAAI/bge-m3' which is a good general-purpose model.
            device: Device to run the model on ('cuda', 'mps', 'cpu'). If None, auto-detects.
        """
        logger.info(f"Loading Sentence Transformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded with embedding dimension: {self.embedding_dimension}")

    def generate_embedding(self, text_list: list[str]) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text_list: List of input texts to generate embeddings for.

        Returns:
            A numpy array containing the text embedding.
        """
        try:
            return self.model.encode(
                text_list, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def generate_embeddings_batch(
        self, documents: List[Document], batch_size: int = 32
    ) -> List[List[float]]:
        """Generate embeddings for a batch of documents.

        Args:
            documents: List of document objects with 'page_content' attribute.
            batch_size: Number of documents to process in each batch.

        Returns:
            List of embeddings.
        """
        if not documents:
            logger.warning("No documents provided for embedding generation")
            return []

        try:
            # Extract text content from documents
            texts = [doc.page_content for doc in documents]

            logger.info(
                f"Generating embeddings for {len(documents)} documents in batches of {batch_size}"
            )

            # Combine documents with their embeddings and metadata
            results = []
            for i in range(0, len(texts), batch_size):
                logger.info(f"Processing batch {i // batch_size + 1}/{len(texts) // batch_size}")
                batch_texts = texts[i : i + batch_size]
                batch_embeddings = self.generate_embedding(batch_texts)
                results.extend(batch_embeddings)
            assert len(results) == len(documents)
            logger.info(f"Completed embedding generation for {len(results)} documents")
            return results

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}", exc_info=True)
            raise

    @property
    def dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.embedding_dimension
