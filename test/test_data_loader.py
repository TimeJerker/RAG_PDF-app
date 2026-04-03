import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import numpy as np
from types import SimpleNamespace
from data_loader import load_and_chunk_pdf, embed_texts

@pytest.fixture
def fake_document():
    docs_1 = SimpleNamespace(text="Это первый русский текст")
    docs_2 = SimpleNamespace(text="This is english text")
    docs_3 = SimpleNamespace()

    return [docs_1, docs_2, docs_3]


@pytest.fixture
def work_in_tmp(tmp_path, monkeypatch):
    """
    Переключаем рабочую директорию на временную.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path

@pytest.fixture
def fake_splitter_chunks():
    return["chunk_1", "chunk_2", "chunk_3"]


@patch("data_loader.PDFReader")
@patch("data_loader.splitter")
def test_load_and_chunk_pdf(mock_splitter, mock_pdf_reader, fake_document, fake_splitter_chunks, work_in_tmp):
    mock_pdf_reader.return_value.load_data.return_value = fake_document

    mock_splitter.split_text.return_value = fake_splitter_chunks

    result_path, result_len = load_and_chunk_pdf("fake_path")

    assert os.path.exists(result_path)

    assert result_path.endswith(".json")

    expected_len = len(fake_splitter_chunks) * 2
    assert result_len == expected_len

    assert mock_splitter.split_text.call_count == 2
    mock_splitter.split_text.assert_any_call("Это первый русский текст")
    mock_splitter.split_text.assert_any_call("This is english text")




@pytest.mark.parametrize("is_query, expected_prefix", [
    (True, "query: "),
    (False, "passage: "),
])
@patch("data_loader.model")
def test_embed_text(mock_model, is_query, expected_prefix):
    fake_text = ["position"]
    fake_encode = np.array([[1,2,3]])

    mock_model.tolist.return_value = fake_encode

    embed_texts(fake_text, is_query=is_query)

    call_args = mock_model.encode.call_args
    actual_text = call_args[0][0]

    assert actual_text == [f"{expected_prefix}position"]


