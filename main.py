import numpy as np
import json
import os
import logging
from typing import List, Dict, Tuple, Union
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textstat
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from rouge_coherence_scorer import ROUGEEnhancedCoherenceScorer


class SummaryEvaluator:
    """
    A comprehensive evaluator for AI-generated summaries that assesses
    accuracy, completeness, and coherence using NLP techniques.
    """

    def __init__(self, genai_model: str, sentence_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the evaluator with required models.

        Args:
            genai_model: OpenAI model name for LLM-as-judge accuracy scoring (e.g., "gpt-4o", "gpt-4.1")
            sentence_model: Sentence transformer model name for semantic similarity
        """
        # Download required NLTK data
        nltk_datasets = [
            ("tokenizers/punkt", "punkt"),
            ("corpora/stopwords", "stopwords"),
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
            ("chunkers/maxent_ne_chunker", "maxent_ne_chunker"),
            ("corpora/words", "words"),
        ]

        for resource, download_name in nltk_datasets:
            try:
                nltk.data.find(resource)
            except LookupError:
                nltk.download(download_name, quiet=True)

        # Initialize models
        self.genai_model = genai_model
        self.sentence_model = SentenceTransformer(sentence_model)
        self.stop_words = set(stopwords.words("english"))
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.rouge_coherence_scorer = ROUGEEnhancedCoherenceScorer(sentence_model)

    def extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text using sentence segmentation."""
        sentences = sent_tokenize(text)
        # Filter out very short sentences that are unlikely to contain substantial claims
        claims = [s.strip() for s in sentences if len(s.split()) > 3]
        return claims

    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text using NLTK."""
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)

        try:
            # Use NLTK's named entity chunker
            tree = ne_chunk(pos_tags)
            entities = []

            for subtree in tree:
                if isinstance(subtree, Tree):
                    # Named entity found
                    entity_name = " ".join([token for token, pos in subtree.leaves()])
                    entities.append(entity_name.lower())
                elif subtree[1] in ["NNP", "NNPS"]:
                    # Fallback: include proper nouns
                    entities.append(subtree[0].lower())

            return entities
        except (LookupError, AttributeError, ValueError, TypeError) as e:
            # Handle expected exceptions from NLTK chunker or data processing
            # LookupError: Missing NLTK data
            # AttributeError: Unexpected data structure
            # ValueError/TypeError: Invalid input data
            entities = [
                word.lower() for word, pos in pos_tags if pos in ["NNP", "NNPS"]
            ]
            return entities
        except Exception as e:
            # Re-raise unexpected exceptions
            raise RuntimeError(f"Unexpected error in extract_entities: {e}") from e

    def find_contradictions(self, claims: List[str], threshold: float = -0.2) -> int:
        """
        Find potential contradictions within a list of claims.

        Args:
            claims: List of textual claims
            threshold: Similarity threshold below which claims are considered contradictory

        Returns:
            Number of contradictory claim pairs
        """
        if len(claims) < 2:
            return 0

        embeddings = self.sentence_model.encode(claims)
        similarities = cosine_similarity(embeddings)

        contradictions = 0
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                if similarities[i][j] < threshold:
                    contradictions += 1

        return contradictions

    def accuracy_score(
        self,
        ai_summary: str,
        source_text: str,
        debug: bool = False,
        return_rationale: bool = False,
    ) -> Union[float, Tuple[float, str]]:
        """
        Calculate accuracy score using LLM-as-judge with GPT-4.

        Args:
            ai_summary: AI-generated summary to evaluate
            source_text: Original source material
            debug: If True, log rationale to console/logger at DEBUG level
            return_rationale: If True, return (score, rationale) tuple instead of just score

        Returns:
            float: Accuracy score between 0 and 1, OR
            Tuple[float, str]: (score, rationale) if return_rationale=True

        Score Ranges:
            1.0: Excellent (3/3) - Highly accurate, well-aligned with source
            0.67: Good (2/3) - Generally accurate with minor issues
            0.33: Fair (1/3) - Some accuracy problems or contradictions
            0.0: Poor (0/3) - Significant inaccuracies or contradictions
        """
        if not ai_summary.strip() or not source_text.strip():
            score = 0.0
            rationale = "Empty summary or source text provided"

            if debug:
                logging.debug(f"Accuracy Score: {score:.3f} | Rationale: {rationale}")

            return (score, rationale) if return_rationale else score

        prompt = f"""Compare the <source> and <summary> provided. Using only the information provided in the <source>, 
characterize the accuracy of the <summary> on a scale of 0=Poor, 1=Fair, 2=Good, or 3=Excellent. 
Provide a short rationale for your score. Format the response into a json document with the following schema
<schema>
{{\n\"score\": int,\n\"rationale\": str\n}}
</schema

<source>
{source_text}
</source>

<summary>
{ai_summary}
</summary>"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.genai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator of text summaries. Your task is to assess the accuracy of summaries based solely on the provided source material.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=200,
            )

            response_content = response.choices[0].message.content.strip()

            # Try to parse JSON response (handle markdown code blocks)
            try:
                # Remove markdown code block markers if present
                if response_content.startswith("```json"):
                    response_content = (
                        response_content.replace("```json", "")
                        .replace("```", "")
                        .strip()
                    )
                elif response_content.startswith("```"):
                    response_content = response_content.replace("```", "").strip()

                result = json.loads(response_content)
                score = result.get("score", 0)
                rationale = result.get("rationale", "No rationale provided")
            except json.JSONDecodeError:
                # Fallback: try to extract score from text
                import re

                score_match = re.search(r"(\d+)", response_content)
                score = int(score_match.group(1)) if score_match else 0
                rationale = (
                    f"Failed to parse JSON response: {response_content[:100]}..."
                )

            # Normalize score to 0-1 range (GPT-4 returns 0-3 scale)
            normalized_score = min(1.0, max(0.0, score / 3.0))

            if debug:
                logging.debug(
                    f"Accuracy Score: {normalized_score:.3f} | Rationale: {rationale}"
                )

            return (
                (normalized_score, rationale) if return_rationale else normalized_score
            )

        except Exception as e:
            # Raise exception if OpenAI API fails - no fallback
            raise RuntimeError(f"LLM-as-judge accuracy scoring failed: {e}") from e

    def extract_important_topics(
        self, text: str, n_topics: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Extract important topics from text using TF-IDF and position weighting.

        Args:
            text: Input text
            n_topics: Number of top topics to extract

        Returns:
            List of (topic, importance_score) tuples
        """
        sentences = sent_tokenize(text)

        if len(sentences) < 2:
            return [(text, 1.0)]

        # Calculate TF-IDF scores
        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Calculate sentence importance scores
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            # Position weight (first and last sentences get higher weight)
            if i == 0 or i == len(sentences) - 1:
                position_weight = 1.5
            elif i < len(sentences) * 0.2 or i > len(sentences) * 0.8:
                position_weight = 1.2
            else:
                position_weight = 1.0

            # TF-IDF weight
            tfidf_score = np.mean(tfidf_matrix[i].toarray())

            # Length weight (longer sentences often more important)
            length_weight = min(1.5, len(sentence.split()) / 20)

            total_score = position_weight * tfidf_score * length_weight
            sentence_scores.append((sentence, total_score))

        # Sort by importance and return top n
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        return sentence_scores[:n_topics]

    def completeness_score(self, ai_summary: str, source_text: str) -> float:
        """
        Calculate completeness score based on coverage of important topics.

        Args:
            ai_summary: AI-generated summary to evaluate
            source_text: Original source material

        Returns:
            Completeness score between 0 and 1

        Score Ranges:
            0.8-1.0: Excellent - Comprehensive coverage of key topics
            0.6-0.8: Good - Covers most important topics adequately
            0.4-0.6: Fair - Covers some topics but misses important content
            0.0-0.4: Poor - Significant gaps in topic coverage
        """
        # Extract important topics from source
        important_topics = self.extract_important_topics(source_text)

        if not important_topics:
            return 1.0

        # Get embeddings for summary and important topics
        summary_embedding = self.sentence_model.encode([ai_summary])
        topic_texts = [topic[0] for topic in important_topics]
        topic_embeddings = self.sentence_model.encode(topic_texts)

        # Calculate coverage
        similarities = cosine_similarity(summary_embedding, topic_embeddings)[0]
        coverage_threshold = 0.4

        covered_topics = 0
        total_importance = 0
        covered_importance = 0

        for i, (topic, importance) in enumerate(important_topics):
            total_importance += importance
            if similarities[i] >= coverage_threshold:
                covered_topics += 1
                covered_importance += importance

        # Calculate coverage metrics
        coverage_recall = covered_topics / len(important_topics)
        importance_weighting = (
            covered_importance / total_importance if total_importance > 0 else 0
        )

        # More balanced formula combining recall and importance weighting
        completeness = min(1.0, (coverage_recall * 0.7 + importance_weighting * 0.3))
        return completeness

    def get_discourse_markers(self) -> Dict[str, List[str]]:
        """
        Get discourse markers based on Penn Discourse Treebank (PDTB) categories.
        This provides a more linguistically grounded approach than hard-coded lists.
        """
        return {
            "temporal": [
                "then",
                "next",
                "meanwhile",
                "subsequently",
                "previously",
                "earlier",
                "later",
                "finally",
                "eventually",
                "afterward",
                "before",
                "after",
                "during",
                "while",
                "until",
                "once",
                "now",
                "currently",
            ],
            "contingency": [
                "because",
                "since",
                "as",
                "therefore",
                "thus",
                "consequently",
                "as a result",
                "due to",
                "so",
                "hence",
                "accordingly",
                "for this reason",
                "given that",
                "in order to",
                "so that",
                "if",
                "unless",
                "provided that",
            ],
            "comparison": [
                "however",
                "but",
                "nevertheless",
                "nonetheless",
                "yet",
                "still",
                "whereas",
                "while",
                "in contrast",
                "on the other hand",
                "conversely",
                "alternatively",
                "instead",
                "rather",
                "similarly",
                "likewise",
                "in the same way",
                "by comparison",
                "although",
                "though",
                "despite",
            ],
            "expansion": [
                "and",
                "also",
                "furthermore",
                "moreover",
                "additionally",
                "in addition",
                "besides",
                "for example",
                "for instance",
                "specifically",
                "in particular",
                "namely",
                "that is",
                "in other words",
                "indeed",
                "in fact",
                "actually",
            ],
        }

    def _count_discourse_markers(self, sentences, discourse_markers):
        """Count discourse markers by category and total markers in sentences."""
        import re

        def create_marker_pattern(marker):
            """Create regex pattern for exact word/phrase matching."""
            return r"\b" + re.escape(marker) + r"\b"

        def find_first_marker_in_sentence(sentence_lower, discourse_markers):
            """Find the first discourse marker in a sentence, return category or None."""
            for category, markers in discourse_markers.items():
                for marker in markers:
                    pattern = create_marker_pattern(marker)
                    if re.search(pattern, sentence_lower):
                        return category
            return None

        category_counts = dict.fromkeys(discourse_markers.keys(), 0)
        total_markers = 0

        for sentence in sentences:
            sentence_lower = sentence.lower()
            category = find_first_marker_in_sentence(sentence_lower, discourse_markers)

            if category:
                category_counts[category] += 1
                total_markers += 1

        return category_counts, total_markers

    def analyze_discourse_markers(self, text: str) -> float:
        """
        Analyze the presence and quality of discourse markers using PDTB categories.

        Args:
            text: Input text to analyze

        Returns:
            Discourse marker score (0-1) with category-based weighting
        """
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 1.0

        discourse_markers = self.get_discourse_markers()
        all_markers = []
        for category_markers in discourse_markers.values():
            all_markers.extend(category_markers)

        # Count markers by category for more nuanced scoring
        category_counts, total_markers = self._count_discourse_markers(
            sentences, discourse_markers
        )

        # Calculate diversity bonus (using different types of markers is better)
        categories_used = sum(1 for count in category_counts.values() if count > 0)
        diversity_bonus = min(1.0, categories_used / len(discourse_markers))

        # Base score: proportion of sentences with discourse markers
        transitions = len(sentences) - 1
        base_score = total_markers / transitions if transitions > 0 else 1.0

        # Combined score with diversity consideration
        discourse_score = min(1.0, (0.8 * base_score + 0.2 * diversity_bonus))

        return discourse_score

    def calculate_logical_flow(self, text: str) -> float:
        """Calculate logical flow based on sentence-to-sentence coherence."""
        sentences = sent_tokenize(text)

        if len(sentences) < 2:
            return 1.0

        embeddings = self.sentence_model.encode(sentences)

        # Calculate consecutive sentence similarities
        flow_scores = []
        for i in range(len(sentences) - 1):
            similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            flow_scores.append(max(0, similarity))  # Ensure non-negative

        logical_flow = np.mean(flow_scores)
        return logical_flow

    def coherence_score(self, ai_summary: str) -> float:
        """
        Calculate coherence score using ROUGE enhanced coherence scoring.

        Args:
            ai_summary: AI-generated summary to evaluate

        Returns:
            Coherence score between 0 and 1

        Score Ranges:
            0.7-1.0: Excellent - Clear, well-structured, and logically flowing
            0.5-0.7: Good - Generally coherent with minor flow issues
            0.3-0.5: Fair - Some coherence problems, may be too short/vague
            0.0-0.3: Poor - Incoherent, fragmented, or very low quality
        """
        return self.rouge_coherence_scorer.score(ai_summary)

    def generate_summary(
        self,
        source_text: str,
        quality: str = "good",
        length: str = "short",
    ) -> str:
        """
        Generate an AI summary of the given source text using OpenAI API.

        Args:
            source_text: Original text to summarize
            quality: Quality level - "poor", "good", or "excellent"
            length: Length preference - "short" or "long"

        Returns:
            Generated summary string

        Raises:
            ValueError: If quality or length parameters are invalid
            RuntimeError: If OpenAI API call fails
        """
        if quality not in ["poor", "good", "excellent"]:
            raise ValueError(
                f"Quality must be 'poor', 'good', or 'excellent', got: {quality}"
            )

        if length not in ["short", "long"]:
            raise ValueError(f"Length must be 'short' or 'long', got: {length}")

        # Define quality-specific instructions
        quality_instructions = {
            "poor": "Write a vague, overly brief summary that misses key points and uses unclear language. Include some inaccuracies or contradicting oversimplifications.",
            "good": "Write a clear, accurate summary that captures the main points effectively. Maintain good coherence and appropriate detail level. Use only the information from the source article.",
            "excellent": "Write an outstanding summary that comprehensively covers all key points with perfect accuracy, excellent flow, and optimal detail balance. Use only the information from the source article.",
        }

        # Define length-specific instructions
        length_instructions = {
            "short": "Keep the summary concise, around 2-4 sentences.",
            "long": "Provide a more detailed summary, around 6-10 sentences with comprehensive coverage.",
        }

        prompt = f"""Please summarize the following text according to these specifications:

Quality Level: {quality.title()}
- {quality_instructions[quality]}

Length: {length.title()}
- {length_instructions[length]}

Text to summarize:
{source_text}

Generate only the summary, no additional commentary."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.genai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert text summarizer. Follow the given quality and length specifications exactly.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3 if quality == "excellent" else 0.7,
                max_tokens=400 if length == "long" else 150,
            )

            summary = response.choices[0].message.content.strip()
            return summary

        except Exception as e:
            raise RuntimeError(f"Summary generation failed: {e}") from e

    def evaluate_summary(
        self,
        ai_summary: str,
        source_text: str,
        weights: Dict[str, float] = None,
        debug: bool = False,
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of AI summary across all dimensions.

        Args:
            ai_summary: AI-generated summary to evaluate
            source_text: Original source material
            weights: Optional weights for combining scores
            debug: If True, enable debug logging for accuracy scoring

        Returns:
            Dictionary with individual and overall scores

        Overall Score Ranges:
            0.7-1.0: Excellent - High-quality summary across all dimensions
            0.5-0.7: Good - Generally solid summary with minor weaknesses
            0.3-0.5: Fair - Acceptable but has notable issues in one or more areas
            0.0-0.3: Poor - Significant problems, needs major improvement
        """
        if weights is None:
            weights = {"accuracy": 0.6, "completeness": 0.25, "coherence": 0.15}

        # Calculate individual scores
        if debug:
            # Get accuracy score with rationale when debug is enabled
            accuracy, accuracy_rationale = self.accuracy_score(
                ai_summary, source_text, debug=debug, return_rationale=True
            )
        else:
            # Get accuracy score only when debug is disabled
            accuracy = self.accuracy_score(ai_summary, source_text, debug=debug)
            accuracy_rationale = None

        completeness = self.completeness_score(ai_summary, source_text)
        coherence = self.coherence_score(ai_summary)

        # Calculate overall score
        overall = (
            weights["accuracy"] * accuracy
            + weights["completeness"] * completeness
            + weights["coherence"] * coherence
        )

        # Build results dictionary
        results = {
            "accuracy": accuracy,
            "completeness": completeness,
            "coherence": coherence,
            "overall": overall,
        }

        # Add rationale to results if available (debug mode)
        if accuracy_rationale is not None:
            results["accuracy_rationale"] = accuracy_rationale

        return results


# Test Suite
import unittest


class TestSummaryEvaluator(unittest.TestCase):
    """Test suite for SummaryEvaluator class."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        cls.evaluator = SummaryEvaluator("gpt-4.1")

        # Sample texts for testing
        cls.source_text = """
        Climate change is one of the most pressing issues of our time. Rising global temperatures
        are causing widespread environmental impacts including melting ice caps, rising sea levels,
        and more frequent extreme weather events. The primary cause is increased greenhouse gas
        emissions from human activities, particularly carbon dioxide from burning fossil fuels.
        Scientists agree that immediate action is needed to reduce emissions and transition to
        renewable energy sources. However, economic and political challenges make implementation
        difficult. International cooperation through agreements like the Paris Climate Accord
        represents important progress, but more ambitious targets are needed.
        """

        cls.good_summary = """
        Climate change is a major global issue caused by rising temperatures from greenhouse gas emissions.
        It's causing environmental problems like ice melting and sea level rise. Scientists say we need to
        reduce emissions and use renewable energy, but there are economic and political challenges.
        """

        cls.poor_summary = """
        Weather is changing. Ice is melting somewhere. People are worried about something.
        There are some agreements about climate stuff.
        """

        cls.inaccurate_summary = """
        Climate change is actually beneficial for the environment, causing ice caps to grow larger
        and making weather more stable. Scientists disagree about the causes, with many believing
        natural cycles are responsible rather than human activities.
        """

        # Additional test case for excellent summary
        cls.ai_source_text = """
        Artificial intelligence has made remarkable progress in recent years, with applications
        spanning from healthcare to transportation. Machine learning algorithms can now diagnose
        diseases, drive cars, and even create art. However, these advances also raise important
        ethical questions about privacy, job displacement, and algorithmic bias. Researchers
        and policymakers are working to develop frameworks for responsible AI development.
        """

        cls.excellent_summary = """
        Climate change, driven largely by human greenhouse gas emissions, is causing melting ice,
        rising seas, and more extreme weather. Scientists agree urgent action is needed to cut
        emissions and shift to renewable energy, but political and economic hurdles slow progress.
        International efforts like the Paris Accord help, though stronger commitments are still required.
        """

    def test_accuracy_score_good_summary(self):
        """Test accuracy scoring with a good summary."""
        score = self.evaluator.accuracy_score(self.good_summary, self.source_text)
        self.assertGreater(score, 0.6, "Good summary should have high accuracy score")
        self.assertLessEqual(score, 1.0, "Accuracy score should not exceed 1.0")

    def test_accuracy_score_inaccurate_summary(self):
        """Test accuracy scoring with an inaccurate summary."""
        score = self.evaluator.accuracy_score(self.inaccurate_summary, self.source_text)
        self.assertLess(
            score, 0.55, "Inaccurate summary should have low accuracy score"
        )

    def test_accuracy_score_excellent_summary(self):
        """Test accuracy scoring with an excellent summary."""
        score = self.evaluator.accuracy_score(self.excellent_summary, self.source_text)
        self.assertGreaterEqual(
            score, 0.8, "Excellent summary should have very high accuracy score"
        )
        self.assertLessEqual(score, 1.0, "Accuracy score should not exceed 1.0")

    def test_completeness_score_good_summary(self):
        """Test completeness scoring with a good summary."""
        score = self.evaluator.completeness_score(self.good_summary, self.source_text)
        self.assertGreater(
            score, 0.6, "Good summary should have reasonable completeness score"
        )

    def test_completeness_score_poor_summary(self):
        """Test completeness scoring with a poor summary."""
        score = self.evaluator.completeness_score(self.poor_summary, self.source_text)
        self.assertLess(score, 0.6, "Poor summary should have low completeness score")

    def test_coherence_score_good_summary(self):
        """Test coherence scoring with a good summary."""
        score = self.evaluator.coherence_score(self.good_summary)
        self.assertGreater(
            score, 0.5, "Good summary should have reasonable coherence score"
        )

    def test_coherence_score_poor_summary(self):
        """Test coherence scoring with a poor summary."""
        score = self.evaluator.coherence_score(self.poor_summary)
        self.assertLess(
            score,
            0.5,
            "Poor summary should have low coherence score due to vagueness and inappropriate length",
        )

    def test_evaluate_summary_comprehensive(self):
        """Test comprehensive evaluation function."""
        results = self.evaluator.evaluate_summary(self.good_summary, self.source_text)

        # Check that all expected keys are present
        expected_keys = ["accuracy", "completeness", "coherence", "overall"]
        for key in expected_keys:
            self.assertIn(key, results, f"Results should contain {key}")

        # Check that all scores are in valid range
        for key, score in results.items():
            self.assertGreaterEqual(score, 0.0, f"{key} score should be >= 0")
            self.assertLessEqual(score, 1.0, f"{key} score should be <= 1")

    def test_extract_claims(self):
        """Test claim extraction functionality."""
        claims = self.evaluator.extract_claims(self.source_text)
        self.assertGreater(len(claims), 0, "Should extract at least one claim")
        self.assertTrue(
            all(len(claim.split()) > 3 for claim in claims),
            "All claims should have more than 3 words",
        )

    def test_extract_important_topics(self):
        """Test topic extraction functionality."""
        topics = self.evaluator.extract_important_topics(self.source_text, n_topics=5)
        self.assertLessEqual(
            len(topics), 5, "Should not exceed requested number of topics"
        )
        self.assertTrue(
            all(isinstance(score, (int, float)) for _, score in topics),
            "All topic scores should be numeric",
        )

    def test_score_bounds(self):
        """Test that all scores remain within bounds for various inputs."""
        test_cases = [
            ("", self.source_text),  # Empty summary
            (self.good_summary, ""),  # Empty source
            ("Single sentence.", "Another single sentence."),  # Minimal input
        ]

        for summary, source in test_cases:
            if summary and source:  # Skip invalid combinations
                results = self.evaluator.evaluate_summary(summary, source)
                for metric, score in results.items():
                    self.assertGreaterEqual(
                        score,
                        0.0,
                        f"{metric} should be >= 0 for case: {summary[:20]}...",
                    )
                    self.assertLessEqual(
                        score,
                        1.0,
                        f"{metric} should be <= 1 for case: {summary[:20]}...",
                    )

    def test_generate_summary_valid_params(self):
        """Test summary generation with valid parameters."""
        # Test all valid combinations
        for quality in ["poor", "good", "excellent"]:
            for length in ["short", "long"]:
                try:
                    summary = self.evaluator.generate_summary(
                        self.source_text, quality=quality, length=length
                    )
                    self.assertIsInstance(
                        summary, str, "Generated summary should be a string"
                    )
                    self.assertGreater(
                        len(summary.strip()), 0, "Generated summary should not be empty"
                    )
                except Exception as e:
                    self.fail(f"Summary generation failed for {quality}/{length}: {e}")

    def test_generate_summary_invalid_params(self):
        """Test summary generation with invalid parameters."""
        # Test invalid quality
        with self.assertRaises(ValueError):
            self.evaluator.generate_summary(self.source_text, quality="invalid")

        # Test invalid length
        with self.assertRaises(ValueError):
            self.evaluator.generate_summary(self.source_text, length="invalid")

    def test_generate_summary_different_lengths(self):
        """Test that different length settings produce appropriately sized summaries."""
        short_summary = self.evaluator.generate_summary(
            self.source_text, quality="good", length="short"
        )
        long_summary = self.evaluator.generate_summary(
            self.source_text, quality="good", length="long"
        )

        # Long summaries should generally be longer than short ones
        # (though this isn't guaranteed, it's a reasonable expectation)
        short_words = len(short_summary.split())
        long_words = len(long_summary.split())

        self.assertGreater(short_words, 0, "Short summary should have words")
        self.assertGreater(long_words, 0, "Long summary should have words")

    @classmethod
    def report_test_scores(cls):
        """Report scores for test summaries to demonstrate evaluation behavior."""
        evaluator = SummaryEvaluator("gpt-4.1")

        print("\n" + "=" * 60)
        print("TEST SUMMARY EVALUATION SCORES")
        print("=" * 60)

        # Excellent summary scores
        excellent_results = evaluator.evaluate_summary(
            cls.excellent_summary, cls.source_text
        )
        print("\nEXCELLENT SUMMARY SCORES:")
        print(f"  Accuracy:    {excellent_results['accuracy']:.3f}")
        print(f"  Completeness: {excellent_results['completeness']:.3f}")
        print(f"  Coherence:   {excellent_results['coherence']:.3f}")
        print(f"  Overall:     {excellent_results['overall']:.3f}")

        # Good summary scores
        good_results = evaluator.evaluate_summary(cls.good_summary, cls.source_text)
        print("\nGOOD SUMMARY SCORES:")
        print(f"  Accuracy:    {good_results['accuracy']:.3f}")
        print(f"  Completeness: {good_results['completeness']:.3f}")
        print(f"  Coherence:   {good_results['coherence']:.3f}")
        print(f"  Overall:     {good_results['overall']:.3f}")

        # Poor summary scores
        poor_results = evaluator.evaluate_summary(cls.poor_summary, cls.source_text)
        print("\nPOOR SUMMARY SCORES:")
        print(f"  Accuracy:    {poor_results['accuracy']:.3f}")
        print(f"  Completeness: {poor_results['completeness']:.3f}")
        print(f"  Coherence:   {poor_results['coherence']:.3f}")
        print(f"  Overall:     {poor_results['overall']:.3f}")

        # Inaccurate summary scores
        inaccurate_results = evaluator.evaluate_summary(
            cls.inaccurate_summary, cls.source_text
        )
        print("\nINACCURATE SUMMARY SCORES:")
        print(f"  Accuracy:    {inaccurate_results['accuracy']:.3f}")
        print(f"  Completeness: {inaccurate_results['completeness']:.3f}")
        print(f"  Coherence:   {inaccurate_results['coherence']:.3f}")
        print(f"  Overall:     {inaccurate_results['overall']:.3f}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    # Example usage
    evaluator = SummaryEvaluator("gpt-4o")

    # Sample source text
    source = """
    Artificial intelligence has made remarkable progress in recent years, with applications
    spanning from healthcare to transportation. Machine learning algorithms can now diagnose
    diseases, drive cars, and even create art. However, these advances also raise important
    ethical questions about privacy, job displacement, and algorithmic bias. Researchers
    and policymakers are working to develop frameworks for responsible AI development.
    """

    # Demonstrate summary generation with different quality and length settings
    print("Generated Summaries:")
    print("=" * 50)

    # Generate different types of summaries
    quality_length_combinations = [
        ("excellent", "short"),
        ("good", "short"),
        ("poor", "short"),
        ("excellent", "long"),
        ("good", "long"),
        ("poor", "long"),
    ]

    for quality, length in quality_length_combinations:
        try:
            generated_summary = evaluator.generate_summary(
                source, quality=quality, length=length
            )
            print(f"\n{quality.upper()} {length.upper()} SUMMARY:")
            print(f"'{generated_summary}'")

            # Evaluate the generated summary
            results = evaluator.evaluate_summary(generated_summary, source, debug=False)
            print(
                f"Scores - Accuracy: {results['accuracy']:.3f}, "
                f"Completeness: {results['completeness']:.3f}, "
                f"Coherence: {results['coherence']:.3f}, "
                f"Overall: {results['overall']:.3f}"
            )
        except Exception as e:
            print(f"\nError generating {quality} {length} summary: {e}")

    # Run tests
    print("\nRunning tests...")
    unittest.main(argv=[""], exit=False, verbosity=2)

    # Report test summary scores
    TestSummaryEvaluator.report_test_scores()
