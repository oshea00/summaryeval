import numpy as np
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree
from sklearn.metrics.pairwise import cosine_similarity
import textstat
from sentence_transformers import SentenceTransformer
import re
from rouge_score import rouge_scorer


class ROUGEEnhancedCoherenceScorer:
    """
    ROUGE enhanced coherence scorer that provides comprehensive coherence evaluation
    by combining traditional coherence metrics with ROUGE-based n-gram analysis.

    Uses the official rouge-score library for accurate ROUGE-1, ROUGE-2, and ROUGE-L calculations.

    Score Normalization:
        All scores are normalized to the range [0.0, 1.0] where:
        - 0.7-1.0: Excellent coherence - Clear, well-structured, logically flowing
        - 0.5-0.7: Good coherence - Generally coherent with minor flow issues
        - 0.3-0.5: Fair coherence - Some coherence problems, may be too short/vague
        - 0.0-0.3: Poor coherence - Incoherent, fragmented, or very low quality

    Component Metrics (all normalized 0.0-1.0):
        - ROUGE-1/2/L: Official F1 scores using rouge-score library with stemming
        - Semantic coherence: Sentence embedding similarities combined with ROUGE-L
        - Discourse coherence: Penn Discourse Treebank marker analysis
        - Lexical diversity: Type-Token Ratio with windowed normalization
        - Contradiction penalty: 1.0 (no contradictions) to 0.0 (many contradictions)
        - Readability: Flesch-Kincaid grade normalized as max(0, min(1, (20-grade)/20))
        - Length penalty: 1.0 (normal length) to 0.3 minimum (very short texts)

    Final Score Calculation:
        Weighted combination: 0.40×semantic + 0.25×discourse + 0.08×(rouge1+rouge2+rougel+lexical) + 0.03×readability
        Then multiplied by contradiction_penalty × length_penalty and clamped to [0.0, 1.0]
    """

    def __init__(self, sentence_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the ROUGE enhanced coherence scorer.

        Args:
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

        self.sentence_model = SentenceTransformer(sentence_model)
        self.stop_words = set(stopwords.words("english"))
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def calculate_rouge_l(self, summary: str, reference: str = None) -> float:
        """
        Calculate ROUGE-L for lexical coherence using rouge-score library.

        Args:
            summary: The summary text to analyze
            reference: Optional reference text. If None, analyzes internal coherence

        Returns:
            ROUGE-L F1 score
        """
        if reference is None:
            sentences = sent_tokenize(summary)
            if len(sentences) < 2:
                return 1.0

            # Use leave-one-out approach with proper ROUGE-L scoring
            coherence_scores = []

            for i, target_sentence in enumerate(sentences):
                # Create reference from all other sentences
                other_sentences = [
                    sentences[j] for j in range(len(sentences)) if j != i
                ]
                if not other_sentences:
                    coherence_scores.append(0.5)
                    continue

                reference_text = " ".join(other_sentences)
                scores = self.rouge_scorer.score(reference_text, target_sentence)
                rouge_l_score = scores["rougeL"].fmeasure
                coherence_scores.append(rouge_l_score)

            # Return average coherence score with scaling
            avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
            return min(
                1.0, avg_coherence * 1.5
            )  # Moderate scaling for internal coherence
        else:
            scores = self.rouge_scorer.score(reference, summary)
            return scores["rougeL"].fmeasure

    def calculate_rouge_n(self, summary: str, n: int = 2) -> float:
        """
        Calculate ROUGE-N for internal n-gram consistency using rouge-score library.

        Args:
            summary: The summary text to analyze
            n: N-gram size (1 for unigrams, 2 for bigrams, etc.)

        Returns:
            ROUGE-N F1 score for internal consistency
        """
        sentences = sent_tokenize(summary)
        if len(sentences) < 2:
            return 1.0

        # Use leave-one-out approach with proper ROUGE-N scoring
        rouge_metric = (
            f"rouge{n}" if n <= 2 else "rouge2"
        )  # Fall back to rouge2 for n > 2
        coherence_scores = []

        for i, target_sentence in enumerate(sentences):
            # Create reference from all other sentences
            other_sentences = [sentences[j] for j in range(len(sentences)) if j != i]
            if not other_sentences:
                coherence_scores.append(0.5)
                continue

            reference_text = " ".join(other_sentences)
            scores = self.rouge_scorer.score(reference_text, target_sentence)
            rouge_n_score = scores[rouge_metric].fmeasure
            coherence_scores.append(rouge_n_score)

        # Calculate average with appropriate scaling for coherence
        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.0

        # Add bonus for optimal n-gram repetition patterns
        all_text = " ".join(sentences)
        all_words = all_text.lower().split()
        if len(all_words) > n:
            unique_ngrams = len(
                {" ".join(all_words[i : i + n]) for i in range(len(all_words) - n + 1)}
            )
            total_ngrams = len(all_words) - n + 1
            repetition_ratio = 1.0 - (unique_ngrams / total_ngrams)

            # Optimal repetition depends on n-gram size
            optimal_repetition = 0.15 if n == 1 else 0.05
            repetition_bonus = 1.0 - abs(repetition_ratio - optimal_repetition) / (
                optimal_repetition + 0.1
            )
            repetition_bonus = max(
                0.0, min(0.3, repetition_bonus * 0.3)
            )  # Limit bonus to 0.3

            final_score = avg_coherence + repetition_bonus
        else:
            final_score = avg_coherence

        return min(1.0, max(0.0, final_score))

    def calculate_lexical_diversity(self, text: str) -> float:
        """
        Calculate lexical diversity (Type-Token Ratio) with ROUGE considerations.

        Args:
            text: Input text

        Returns:
            Lexical diversity score (0-1)
        """
        tokens = word_tokenize(text.lower())
        filtered_tokens = [
            token
            for token in tokens
            if token.isalnum() and token not in self.stop_words
        ]

        if len(filtered_tokens) == 0:
            return 0.0

        unique_tokens = len(set(filtered_tokens))
        total_tokens = len(filtered_tokens)

        # Normalized TTR to handle text length variation
        if total_tokens < 10:
            return unique_tokens / total_tokens
        else:
            # Use moving average TTR for longer texts
            ttr_scores = []
            window_size = min(50, total_tokens // 2)

            for i in range(0, total_tokens - window_size + 1, window_size // 2):
                window_tokens = filtered_tokens[i : i + window_size]
                window_ttr = len(set(window_tokens)) / len(window_tokens)
                ttr_scores.append(window_ttr)

            return np.mean(ttr_scores) if ttr_scores else unique_tokens / total_tokens

    def get_discourse_markers(self) -> Dict[str, List[str]]:
        """
        Get discourse markers based on Penn Discourse Treebank (PDTB) categories.
        Enhanced with ROUGE considerations for coherence analysis.
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
                "initially",
                "ultimately",
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
                "whereas",
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
                "in spite of",
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

    def analyze_discourse_coherence(self, text: str) -> Dict[str, float]:
        """
        Analyze discourse coherence using ROUGE-enhanced discourse marker analysis.

        Args:
            text: Input text

        Returns:
            Dictionary with discourse coherence metrics
        """
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return {
                "marker_diversity": 1.0,
                "marker_density": 1.0,
                "marker_consistency": 1.0,
                "overall_discourse": 1.0,
            }

        discourse_markers = self.get_discourse_markers()

        # Track marker usage
        category_counts = dict.fromkeys(discourse_markers.keys(), 0)
        total_markers = 0
        marker_positions = []

        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            found_marker = False

            for category, markers in discourse_markers.items():
                for marker in markers:
                    pattern = r"\b" + re.escape(marker) + r"\b"
                    if re.search(pattern, sentence_lower) and not found_marker:
                        category_counts[category] += 1
                        total_markers += 1
                        marker_positions.append(i)
                        found_marker = True
                        break
                if found_marker:
                    break

        # Calculate metrics
        transitions_needed = len(sentences) - 1

        # Marker density: proportion of transitions with markers
        marker_density = (
            min(1.0, total_markers / transitions_needed)
            if transitions_needed > 0
            else 1.0
        )

        # Marker diversity: how many different types of markers are used
        categories_used = sum(1 for count in category_counts.values() if count > 0)
        marker_diversity = categories_used / len(discourse_markers)

        # Marker consistency: even distribution of markers throughout text
        if len(marker_positions) < 2:
            marker_consistency = 1.0 if len(marker_positions) == 1 else 0.5
        else:
            # Calculate standard deviation of marker positions
            expected_interval = len(sentences) / len(marker_positions)
            intervals = [
                marker_positions[i + 1] - marker_positions[i]
                for i in range(len(marker_positions) - 1)
            ]
            std_dev = np.std(intervals) if intervals else 0
            marker_consistency = max(0.0, 1.0 - (std_dev / expected_interval))

        # Overall discourse score
        overall_discourse = (
            0.4 * marker_density + 0.3 * marker_diversity + 0.3 * marker_consistency
        )

        return {
            "marker_diversity": marker_diversity,
            "marker_density": marker_density,
            "marker_consistency": marker_consistency,
            "overall_discourse": overall_discourse,
        }

    def calculate_semantic_coherence(self, text: str) -> float:
        """
        Calculate semantic coherence using sentence embeddings and ROUGE-L.

        Args:
            text: Input text

        Returns:
            Semantic coherence score (0-1)
        """
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 1.0

        # Get sentence embeddings
        embeddings = self.sentence_model.encode(sentences)

        # Calculate consecutive sentence similarities
        semantic_scores = []
        for i in range(len(sentences) - 1):
            similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            semantic_scores.append(max(0, similarity))

        # Combine with ROUGE-L for lexical coherence
        rouge_l_score = self.calculate_rouge_l(text)
        semantic_similarity = np.mean(semantic_scores)

        # Weighted combination
        coherence_score = 0.7 * semantic_similarity + 0.3 * rouge_l_score
        return min(1.0, max(0.0, coherence_score))

    def find_contradictions_enhanced(self, text: str, threshold: float = -0.1) -> float:
        """
        Enhanced contradiction detection using ROUGE and semantic analysis.

        Args:
            text: Input text
            threshold: Similarity threshold for contradiction detection

        Returns:
            Contradiction penalty score (0-1, where 1 is no contradictions)
        """
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 1.0

        embeddings = self.sentence_model.encode(sentences)
        similarities = cosine_similarity(embeddings)

        contradictions = 0
        total_pairs = 0

        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                total_pairs += 1

                # Semantic contradiction check
                semantic_sim = similarities[i][j]

                # ROUGE-based lexical contradiction check using proper ROUGE-L
                scores = self.rouge_scorer.score(sentences[i], sentences[j])
                rouge_sim = scores["rougeL"].fmeasure

                # Combined contradiction score
                if semantic_sim < threshold and rouge_sim > 0.3:
                    # High lexical overlap but low semantic similarity suggests contradiction
                    contradictions += 1
                elif semantic_sim < threshold * 1.5:
                    # Very low semantic similarity
                    contradictions += 0.5

        if total_pairs == 0:
            return 1.0

        contradiction_ratio = contradictions / total_pairs
        return max(0.0, 1.0 - contradiction_ratio)

    def calculate_coherence_score(self, text: str) -> Dict[str, float]:
        """
        Calculate comprehensive ROUGE-enhanced coherence score.

        Args:
            text: Text to evaluate for coherence

        Returns:
            Dictionary containing detailed coherence metrics and overall score
        """
        if not text or not text.strip():
            return {
                "rouge_1_coherence": 0.0,
                "rouge_2_coherence": 0.0,
                "rouge_l_coherence": 0.0,
                "semantic_coherence": 0.0,
                "discourse_coherence": 0.0,
                "lexical_diversity": 0.0,
                "contradiction_penalty": 1.0,
                "readability": 0.0,
                "length_penalty": 0.0,
                "overall_coherence": 0.0,
            }

        # Core ROUGE-based metrics
        rouge_1 = self.calculate_rouge_n(text, n=1)
        rouge_2 = self.calculate_rouge_n(text, n=2)
        rouge_l = self.calculate_rouge_l(text)

        # Semantic coherence
        semantic_coh = self.calculate_semantic_coherence(text)

        # Discourse analysis
        discourse_metrics = self.analyze_discourse_coherence(text)
        discourse_coh = discourse_metrics["overall_discourse"]

        # Lexical diversity
        lexical_div = self.calculate_lexical_diversity(text)

        # Contradiction analysis
        contradiction_pen = self.find_contradictions_enhanced(text)

        # Readability (Flesch-Kincaid)
        try:
            fk_grade = textstat.flesch_kincaid_grade(text)
            readability = max(0, min(1, (20 - fk_grade) / 20))
        except Exception:
            readability = 0.5

        # Length penalty for very short summaries
        words = len(text.split())
        length_penalty = 1.0
        if words < 20:
            length_penalty = max(0.3, words / 20)

        # Weighted combination of all metrics (rebalanced for better scores)
        overall_coherence = (
            0.08 * rouge_1  # Unigram coherence
            + 0.08 * rouge_2  # Bigram coherence
            + 0.08 * rouge_l  # Longest common subsequence
            + 0.40 * semantic_coh  # Semantic similarity (primary weight)
            + 0.25 * discourse_coh  # Discourse markers (strong weight)
            + 0.08 * lexical_div  # Lexical diversity
            + 0.03 * readability  # Readability (minimal weight)
        )

        # Apply penalties
        overall_coherence = overall_coherence * contradiction_pen * length_penalty
        overall_coherence = min(1.0, max(0.0, overall_coherence))

        return {
            "rouge_1_coherence": rouge_1,
            "rouge_2_coherence": rouge_2,
            "rouge_l_coherence": rouge_l,
            "semantic_coherence": semantic_coh,
            "discourse_coherence": discourse_coh,
            "lexical_diversity": lexical_div,
            "contradiction_penalty": contradiction_pen,
            "readability": readability,
            "length_penalty": length_penalty,
            "overall_coherence": overall_coherence,
        }

    def score(self, text: str) -> float:
        """
        Main scoring method that returns overall coherence score.
        Compatible with SummaryEvaluator interface.

        Args:
            text: Text to evaluate

        Returns:
            Overall coherence score (0-1)
        """
        result = self.calculate_coherence_score(text)
        return result["overall_coherence"]
