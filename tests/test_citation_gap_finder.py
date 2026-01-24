import unittest
import json
import os
from unittest.mock import MagicMock, patch
from phases.paper_search.citation_gap_finder import CitationGapFinder, SuggestedPaper
from phases.paper_search.paper import Paper
from phases.paper_search.semantic_scholar_api import SemanticScholarAPI

# Path to the real papers file
PAPERS_JSON_PATH = os.path.join(os.path.dirname(__file__), "../output/papers.json")

class TestCitationGapFinder(unittest.TestCase):

    def setUp(self):
        """Set up the finder and mock API for each test."""
        self.mock_s2_api = MagicMock(spec=SemanticScholarAPI)
        self.finder = CitationGapFinder()
        self.finder.s2_api = self.mock_s2_api
        
        # Load sample papers
        if os.path.exists(PAPERS_JSON_PATH):
            with open(PAPERS_JSON_PATH, "r") as f:
                data = json.load(f)
                self.sample_papers = []
                for item in data:
                    self.sample_papers.append(Paper(
                        id=item.get("id", "test_id"),
                        title=item.get("title", "Test Title"),
                        published=item.get("published", "2023"),
                        authors=item.get("authors", ["Test Author"]),
                        summary=item.get("summary", "Test Summary"),
                        pdf_url=item.get("pdf_url"),
                        doi=item.get("doi"),
                        fields_of_study=item.get("fields_of_study", []),
                        venue=item.get("venue"),
                        citation_count=item.get("citation_count", 0)
                    ))
        else:
            self.sample_papers = [
                Paper(
                    id="1",
                    title="Deep Residual Learning for Image Recognition",
                    published="2016",
                    authors=["Kaiming He", "Xiangyu Zhang"],
                    summary="ResNet paper",
                    pdf_url=None,
                    doi=None,
                    fields_of_study=["Computer Science"],
                    venue="CVPR",
                    citation_count=100000
                )
            ]

    def test_identify_missing_papers(self):
        """Test the LLM interaction for identifying missing papers."""
        
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.parsed = {
            "missing_papers": [
                {
                    "title": "Attention Is All You Need",
                    "reason": "Foundational Transformer paper"
                }
            ]
        }
        
        with patch("phases.paper_search.citation_gap_finder.lms.llm") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.respond.return_value = mock_response
            mock_llm_factory.return_value = mock_llm
            
            suggestions = self.finder.identify_missing_papers(
                papers=self.sample_papers,
                research_context="Machine Learning",
                model_name="test-model"
            )
            
            self.assertEqual(len(suggestions), 1)
            self.assertEqual(suggestions[0].title, "Attention Is All You Need")

    def test_search_suggested_papers(self):
        """Test searching for papers using the new match_paper method."""
        
        suggestions = [
            SuggestedPaper(
                title="Attention Is All You Need",
                reason="Foundational Transformer paper"
            )
        ]
        
        existing_ids = set()
        
        # Mock match_paper to return a paper found
        found_paper = Paper(
            id="found_1",
            title="Attention Is All You Need",
            published="2017",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            summary="Transformer architecture",
            pdf_url=None,
            doi=None,
            fields_of_study=["CS"],
            venue="NeurIPS"
        )
        self.mock_s2_api.match_paper.return_value = found_paper
        
        found_papers = self.finder.search_suggested_papers(suggestions, existing_ids)
        
        self.assertEqual(len(found_papers), 1)
        self.assertEqual(found_papers[0].id, "found_1")
        self.assertIn("found_1", existing_ids)
        
        # Verify match_paper was called with correct arguments
        self.mock_s2_api.match_paper.assert_called_with(query="Attention Is All You Need")

    def test_search_suggested_papers_not_found(self):
        """Test when match_paper returns None."""
        
        suggestions = [
            SuggestedPaper(
                title="Non Existent Paper 12345",
                reason="Testing not found"
            )
        ]
        
        existing_ids = set()
        self.mock_s2_api.match_paper.return_value = None
        
        found_papers = self.finder.search_suggested_papers(suggestions, existing_ids)
        
        self.assertEqual(len(found_papers), 0)
        self.mock_s2_api.match_paper.assert_called_with(query="Non Existent Paper 12345")

    def test_search_suggested_papers_duplicate(self):
        """Test that duplicates are filtered out."""
        
        suggestions = [
            SuggestedPaper(title="Paper A", reason="Test")
        ]
        
        existing_ids = {"existing_1"}
        
        # Mock finding a paper that is already in existing_ids
        found_paper = Paper(
            id="existing_1",
            title="Paper A",
            published="2020",
            authors=["Auth A"],
            summary="Duplicate",
            pdf_url=None,
            doi=None,
            fields_of_study=[],
            venue=None
        )
        self.mock_s2_api.match_paper.return_value = found_paper
        
        found_papers = self.finder.search_suggested_papers(suggestions, existing_ids)
        
        self.assertEqual(len(found_papers), 0)

if __name__ == "__main__":
    unittest.main()
