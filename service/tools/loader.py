from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from process.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class Document(BaseModel):
    """A document with page content and metadata."""

    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentLoader:
    """Handles loading and splitting of PDF documents."""

    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 500,
        separators: Optional[List[str]] = None,
    ):
        """Initialize the document loader.

        Args:
            chunk_size: Maximum size of chunks to split documents into
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for splitting text
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load and split a PDF document.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of document chunks with metadata

        Raises:
            FileNotFoundError: If the specified file does not exist
        """
        path = Path(file_path)
        if not path.exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            logger.info(f"Loading PDF: {file_path}")
            # loader = PyPDFLoader(str(path))
            loader = PyMuPDFLoader(str(path))
            pages = loader.load()

            # Convert to our Document format
            documents = []
            for page in pages:
                doc = Document(
                    page_content=page.page_content,
                    metadata={
                        "source": str(path.absolute()),
                        "page": page.metadata.get("page"),
                        "total_pages": len(pages),
                    },
                )
                documents.append(doc)

            logger.info(f"Loaded {len(pages)} pages from {file_path}")
            return documents

        except Exception as e:
            error_msg = f"Error loading PDF {file_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks.

        Args:
            documents: List of documents to split

        Returns:
            List of split document chunks
        """
        if not documents:
            logger.warning("No documents provided to split")
            return []

        split_docs = []
        total_chunks = 0

        logger.info(
            f"Splitting {len(documents)} documents with chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )

        try:
            # Split the document
            splits = self.text_splitter.split_documents(documents)

            # Convert back to our Document format
            for i, split in enumerate(splits):
                # Update metadata with chunk information
                metadata = split.metadata.copy()
                metadata.update(
                    {
                        "chunk_index": i + 1,
                        "total_chunks": len(splits),
                        "split_method": "recursive_character",
                    }
                )

                split_doc = Document(page_content=split.page_content, metadata=metadata)
                split_docs.append(split_doc)

            total_chunks += len(splits)

        except Exception as e:
            source = documents.metadata.get("source", "unknown")
            logger.error(f"Error splitting document {source}: {str(e)}", exc_info=True)

        logger.info(f"Split {len(documents)} documents into {total_chunks} total chunks")
        return split_docs
