"""
Sentiment Analysis Engine Module
Performs multilingual sentiment classification (Urdu, English, Roman Urdu)
with confidence scoring for policy sentiment analysis
"""

from typing import Dict, Tuple
import re


class SentimentAnalyzer:
    """
    Simple sentiment analyzer for Urdu/English/Roman Urdu text.
    Uses rule-based approach with keyword matching and scoring.
    Future: integrate transformer models (UrdueHate, AraBERT, multilingual-BERT)
    """
    
    # Positive sentiment keywords (English)
    POSITIVE_KEYWORDS_EN = {
        "good": 1.0, "great": 1.0, "excellent": 1.0, "amazing": 1.0,
        "love": 1.0, "wonderful": 1.0, "fantastic": 1.0, "perfect": 1.0,
        "best": 1.0, "beautiful": 1.0, "helpful": 0.8, "useful": 0.8,
        "happy": 0.9, "glad": 0.8, "appreciate": 0.9, "thank": 0.7,
        "improve": 0.7, "better": 0.8, "success": 0.9, "benefit": 0.8
    }
    
    # Negative sentiment keywords (English)
    NEGATIVE_KEYWORDS_EN = {
        "bad": 1.0, "terrible": 1.0, "awful": 1.0, "horrible": 1.0,
        "hate": 1.0, "worst": 1.0, "stupid": 1.0, "angry": 0.9,
        "sad": 0.9, "disappointed": 0.9, "fail": 0.9, "problem": 0.7,
        "issue": 0.7, "poor": 0.8, "weak": 0.7, "wrong": 0.8,
        "complain": 0.8, "concern": 0.6, "crisis": 0.9, "corrupt": 1.0
    }
    
    # Positive sentiment keywords (Urdu)
    POSITIVE_KEYWORDS_UR = {
        "اچھا": 1.0, "بہترین": 1.0, "شاندار": 1.0, "خوبصورت": 1.0,
        "محبت": 1.0, "پسند": 0.9, "شکریہ": 0.8, "مدد": 0.7,
        "فائدہ": 0.8, "بہتری": 0.8, "کامیاب": 0.9, "خوشی": 0.9
    }
    
    # Negative sentiment keywords (Urdu)
    NEGATIVE_KEYWORDS_UR = {
        "برا": 1.0, "خراب": 1.0, "بُرا": 1.0, "ناپسند": 0.9,
        "مسئلہ": 0.7, "شکایت": 0.8, "ناکام": 0.9, "پریشان": 0.8,
        "غصہ": 0.9, "ناراض": 0.9, "غلط": 0.8, "مسائل": 0.7
    }
    
    # Intensifiers
    INTENSIFIERS = {
        "very": 1.2, "extremely": 1.3, "really": 1.1, "so": 1.1,
        "absolutely": 1.3, "definitely": 1.2, "surely": 1.1,
        "بہت": 1.2, "انتہائی": 1.3, "یقیناً": 1.2
    }
    
    # Negators
    NEGATORS = ["not", "no", "don't", "doesn't", "didn't", "won't", "can't",
                "نہیں", "نہ", "کوئی"]
    
    def __init__(self):
        """Initialize sentiment analyzer"""
        self.positive_keywords = {**self.POSITIVE_KEYWORDS_EN, **self.POSITIVE_KEYWORDS_UR}
        self.negative_keywords = {**self.NEGATIVE_KEYWORDS_EN, **self.NEGATIVE_KEYWORDS_UR}
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def calculate_sentiment_scores(self, text: str) -> Dict[str, float]:
        """
        Calculate sentiment scores for text.
        
        Args:
            text (str): Text to analyze
        
        Returns:
            dict: Scores for positive, negative, neutral
        """
        text = self.preprocess_text(text)
        words = text.split()
        
        positive_score = 0.0
        negative_score = 0.0
        neutral_score = 0.0
        
        # Track context (negation)
        negation_active = False
        
        for i, word in enumerate(words):
            # Check for negation
            if word in self.NEGATORS:
                negation_active = True
                continue
            
            # Check for intensifiers
            intensifier = 1.0
            if i > 0 and words[i-1] in self.INTENSIFIERS:
                intensifier = self.INTENSIFIERS.get(words[i-1], 1.0)
            
            # Check positive keywords
            if word in self.positive_keywords:
                score = self.positive_keywords[word] * intensifier
                if negation_active:
                    negative_score += score
                else:
                    positive_score += score
                negation_active = False
            
            # Check negative keywords
            elif word in self.negative_keywords:
                score = self.negative_keywords[word] * intensifier
                if negation_active:
                    positive_score += score
                else:
                    negative_score += score
                negation_active = False
        
        # Normalize scores
        total = max(positive_score + negative_score, 1.0)
        positive_score = positive_score / total
        negative_score = negative_score / total
        neutral_score = 1.0 - (positive_score + negative_score)
        
        # Ensure values stay between 0-1
        positive_score = min(positive_score, 1.0)
        negative_score = min(negative_score, 1.0)
        neutral_score = max(0.0, neutral_score)
        
        return {
            "positive": round(positive_score, 3),
            "negative": round(negative_score, 3),
            "neutral": round(neutral_score, 3)
        }
    
    def classify_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Classify text sentiment and return confidence score.
        
        Args:
            text (str): Text to classify
        
        Returns:
            tuple: (sentiment_label, confidence_score)
        """
        scores = self.calculate_sentiment_scores(text)
        
        # Determine sentiment based on max score
        max_score = max(scores.values())
        
        if scores["positive"] == max_score and scores["positive"] > 0.3:
            return ("positive", scores["positive"])
        elif scores["negative"] == max_score and scores["negative"] > 0.3:
            return ("negative", scores["negative"])
        else:
            return ("neutral", scores["neutral"])
    
    def analyze_batch(self, texts, text_field="comment"):
        """
        Analyze sentiment for batch of texts.
        
        Args:
            texts (list): List of dictionaries or strings
            text_field (str): Field name if texts are dicts
        
        Returns:
            list: Updated records with sentiment and confidence
        """
        results = []
        
        for item in texts:
            if isinstance(item, dict):
                text = item.get(text_field, "")
                record = item.copy()
            else:
                text = item
                record = {"comment": text}
            
            sentiment, confidence = self.classify_sentiment(text)
            scores = self.calculate_sentiment_scores(text)
            
            record["sentiment"] = sentiment
            record["sentiment_confidence"] = confidence
            record["positive_score"] = scores["positive"]
            record["negative_score"] = scores["negative"]
            record["neutral_score"] = scores["neutral"]
            
            results.append(record)
        
        return results
    
    def get_sentiment_distribution(self, texts, text_field="comment"):
        """
        Get overall sentiment distribution for batch of texts.
        
        Args:
            texts (list): List of texts to analyze
            text_field (str): Field name if texts are dicts
        
        Returns:
            dict: Sentiment counts and percentages
        """
        analyzed = self.analyze_batch(texts, text_field)
        
        distribution = {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }
        
        for record in analyzed:
            sentiment = record.get("sentiment")
            if sentiment in distribution:
                distribution[sentiment] += 1
        
        total = len(analyzed)
        if total > 0:
            percentages = {
                f"{k}_percent": round((v / total) * 100, 2)
                for k, v in distribution.items()
            }
            distribution.update(percentages)
            distribution["total"] = total
        
        return distribution


# Singleton instance
_analyzer = None


def get_analyzer():
    """Get or create sentiment analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer


def analyze_sentiment(text: str) -> Tuple[str, float]:
    """Convenience function to analyze single text"""
    analyzer = get_analyzer()
    return analyzer.classify_sentiment(text)


def analyze_batch_sentiment(texts, text_field="comment"):
    """Convenience function to analyze batch"""
    analyzer = get_analyzer()
    return analyzer.analyze_batch(texts, text_field)