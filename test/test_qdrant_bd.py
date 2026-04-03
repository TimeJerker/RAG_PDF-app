import pytest
from unittest.mock import patch, MagicMock

from qdrant_bd import QdrantStorage

@patch("qdrant_bd.QdrantClient")
#@patch("qdrant_bd.")
def test_qdrant_search(mock_client):
    
    fake_result = mock_client.query_points.call_args

    fake_vector = [0.2,1.4,2.3]

    storage = QdrantStorage()

    result = storage.search(fake_vector,top_k = 3)

    assert storage
