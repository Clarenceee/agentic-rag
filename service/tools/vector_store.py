from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from utils.logger import get_logger

logger = get_logger(__name__)


class QdrantVectorStore:
    """Handles vector storage and retrieval using Qdrant."""

    def __init__(
        self,
        collection_name: str,
        vector_size: int = 1024,
        distance: str = "Cosine",
        qdrant_url: Optional[str] = None,
    ):
        """Initialize the vector store.

        Args:
            collection_name: Name of the Qdrant collection
            config: Configuration for the vector store
            qdrant_url: URL for the Qdrant server
        """
        self.collection_name = collection_name
        self.distance = distance
        self.vector_size = vector_size
        self.qdrant_url = qdrant_url or "http://localhost:6333"

        # Initialize clients
        logger.info(f"Initializing vector store with url: {self.qdrant_url}")
        self.qdrant_client = QdrantClient(url=self.qdrant_url)

        # Ensure the collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure the Qdrant collection exists, create it if not."""
        collections = self.qdrant_client.get_collections()
        collection_names = [collection.name for collection in collections.collections]

        if self.collection_name not in collection_names:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size, distance=self._get_distance_metric()
                ),
            )
            logger.info(
                f"Collection {self.collection_name} created with distance metric: {self.distance}"
            )
        else:
            logger.info(f"Collection {self.collection_name} already exists")

    def _get_distance_metric(self) -> models.Distance:
        """Get the Qdrant distance metric from config."""
        distance_map = {
            "Cosine": models.Distance.COSINE,
            "Euclid": models.Distance.EUCLID,
            "Dot": models.Distance.DOT,
        }
        return distance_map.get(self.distance, models.Distance.COSINE)

    def add_documents(self, documents, embeddings: List[List[float]]) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of documents
            embeddings: List of embeddings for the documents
        """
        if not documents:
            return

        # Prepare data for upsert
        assert len(documents) == len(embeddings)
        logger.info(f"Adding {len(documents)} documents to vector store")

        points = []
        for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Prepare payload with metadata
            payload = {
                "source": doc.metadata["source"],
                "page": doc.metadata["page"],
                "chunk_index": doc.metadata["chunk_index"],
                "content": doc.page_content,
            }

            # Create point structure
            point = models.PointStruct(
                id=idx,  # Note: 'id' not 'ids'
                vector=embedding,  # Directly use the embedding
                payload=payload,  # Note: 'payload' not 'payloads'
            )
            points.append(point)

            # Upload in batches if needed
            if len(points) >= 100:  # Adjust batch size as needed
                self.qdrant_client.upsert(
                    collection_name=self.collection_name, points=points, wait=True
                )
                points = []

        # Upload any remaining points
        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=points, wait=True
            )

        logger.info(f"Completed adding {len(documents)} documents to vector store")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
        with_content: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.

        Args:
            query_embedding: Search query embedding
            top_k: Number of results to return
            filter_conditions: Optional filter conditions for metadata
            with_content: Whether to include full content in payload

        Returns:
            List of search results with scores and metadata
        """
        # Build filter
        query_filter = None
        if filter_conditions:
            must_conditions = []
            for key, value in filter_conditions.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=f"metadata.{key}", match=models.MatchValue(value=value)
                    )
                )

            if must_conditions:
                query_filter = models.Filter(must=must_conditions)

        # Perform search
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            with_vectors=False,  # Exclude vectors from results
            with_payload=True,  # Include payload in response
        ).points

        return search_result


# docker run -p 6333:6333 -p 6334:6334 \
#     -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
#     qdrant/qdrant
