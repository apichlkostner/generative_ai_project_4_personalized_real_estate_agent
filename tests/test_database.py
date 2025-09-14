import unittest
from unittest.mock import patch, MagicMock
from database import Database

class TestDatabase(unittest.TestCase):
    @patch('database.OpenAIEmbeddings')
    @patch('database.OllamaEmbeddings')
    def test_init_openai(self, mock_ollama, mock_openai):
        db = Database(open_ai=True)
        mock_openai.assert_called_once_with(model="text-embedding-3-small")
        self.assertTrue(hasattr(db, 'embeddings'))

    @patch('database.OpenAIEmbeddings')
    @patch('database.OllamaEmbeddings')
    def test_init_ollama(self, mock_ollama, mock_openai):
        db = Database(open_ai=False)
        mock_ollama.assert_called_once_with(model="mxbai-embed-large")
        self.assertTrue(hasattr(db, 'embeddings'))

    @patch('database.JSONLoader')
    @patch('database.Chroma')
    def test_load_data(self, mock_chroma, mock_jsonloader):
        mock_loader = MagicMock()
        mock_loader.load.return_value = [MagicMock(page_content='content', metadata={'id': 1})]
        mock_jsonloader.return_value = mock_loader
        db = Database(open_ai=True)
        db.embeddings = MagicMock()
        db.load_data('fake.json')
        mock_jsonloader.assert_called_once()
        mock_chroma.from_documents.assert_called_once()
        self.assertTrue(hasattr(db, 'db'))

    @patch('database.Chroma')
    def test_load_db(self, mock_chroma):
        db = Database(open_ai=True)
        db.embeddings = MagicMock()
        db.load_db()
        mock_chroma.assert_called_once()
        self.assertTrue(hasattr(db, 'db'))

    def test_similarity_search(self):
        db = Database(open_ai=True)
        db.db = MagicMock()
        db.db.similarity_search.return_value = ['result1', 'result2']
        results = db.similarity_search('query', k=2)
        db.db.similarity_search.assert_called_once_with('query', 2)
        self.assertEqual(results, ['result1', 'result2'])

if __name__ == '__main__':
    unittest.main()
