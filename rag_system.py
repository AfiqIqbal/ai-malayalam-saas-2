from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from dataclasses import dataclass, field
from loguru import logger

# RAG Configuration
RULES_FILE = "rules_knowledge_base.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight model for local use
SIMILARITY_THRESHOLD = 0.6  # Threshold for rule matching

@dataclass
class Rule:
    """Represents a rule in the knowledge base"""
    id: str
    title: str
    description: str
    content: str
    embedding: Optional[np.ndarray] = None

class RAGSystem:
    """Retrieval-Augmented Generation system for rule-based responses"""
    
    def __init__(self, rules_file: str = RULES_FILE):
        self.rules_file = rules_file
        self.rules: List[Rule] = []
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self._load_rules()
    
    def _load_rules(self) -> None:
        """Load rules from JSON file and generate embeddings"""
        try:
            if os.path.exists(self.rules_file):
                with open(self.rules_file, 'r', encoding='utf-8') as f:
                    rules_data = json.load(f)
                
                self.rules = [
                    Rule(
                        id=rule_data['id'],
                        title=rule_data['title'],
                        description=rule_data.get('description', ''),
                        content=rule_data['content']
                    ) for rule_data in rules_data.get('rules', [])
                ]
                
                # Generate embeddings for all rules
                texts = [f"{rule.title}. {rule.description}" for rule in self.rules]
                if texts:
                    embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
                    for rule, embedding in zip(self.rules, embeddings):
                        rule.embedding = embedding
                
                logger.info(f"Loaded {len(self.rules)} rules from {self.rules_file}")
            else:
                # Create default rules file if it doesn't exist
                default_rules = {
                    "rules": [
                        {
                            "id": "default_rule_1",
                            "title": "Default Behavior",
                            "description": "Default rule when no specific rules match",
                            "content": "You are a helpful AI assistant. Follow these guidelines:\n1. Be polite and respectful\n2. Stay on topic\n3. If you don't know something, say so honestly"
                        }
                    ]
                }
                with open(self.rules_file, 'w', encoding='utf-8') as f:
                    json.dump(default_rules, f, indent=2)
                self.rules = [
                    Rule(
                        id=default_rules['rules'][0]['id'],
                        title=default_rules['rules'][0]['title'],
                        description=default_rules['rules'][0]['description'],
                        content=default_rules['rules'][0]['content'],
                        embedding=self.embedding_model.encode(
                            f"{default_rules['rules'][0]['title']}. {default_rules['rules'][0]['description']}",
                            convert_to_numpy=True
                        )
                    )
                ]
                logger.info(f"Created default rules file at {self.rules_file}")
                
        except Exception as e:
            logger.error(f"Error loading rules: {e}")
            self.rules = []
    
    def get_relevant_rules(self, query: str, top_k: int = 3) -> List[Tuple[Rule, float]]:
        """Find the most relevant rules for a given query"""
        if not self.rules:
            return []
            
        # Get query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        
        # Calculate similarity scores
        similarities = []
        for rule in self.rules:
            if rule.embedding is not None:
                similarity = cosine_similarity(
                    [query_embedding],
                    [rule.embedding]
                )[0][0]
                if similarity >= SIMILARITY_THRESHOLD:
                    similarities.append((rule, similarity))
        
        # Sort by similarity score (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        return similarities[:top_k]
    
    def get_prompt_with_context(self, query: str, max_rules: int = 3) -> str:
        """Create a prompt with relevant rules as context"""
        relevant_rules = self.get_relevant_rules(query, max_rules)
        
        if not relevant_rules:
            # If no rules match, use the first rule as default
            if self.rules:
                rules_text = self.rules[0].content
            else:
                rules_text = "You are a helpful AI assistant. Be polite and helpful."
        else:
            # Combine relevant rules
            rules_text = "\n\n".join(
                f"Rule: {rule.title}\n{rule.content}" 
                for rule, _ in relevant_rules
            )
        
        return f"""Follow these rules when responding:

{rules_text}

User: {query}
Assistant:"""

    def add_rule(self, rule_data: Dict[str, str]) -> bool:
        """Add a new rule to the knowledge base"""
        try:
            # Generate embedding for the new rule
            text = f"{rule_data['title']}. {rule_data.get('description', '')}"
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            
            # Create new rule
            new_rule = Rule(
                id=rule_data['id'],
                title=rule_data['title'],
                description=rule_data.get('description', ''),
                content=rule_data['content'],
                embedding=embedding
            )
            
            # Add to rules
            self.rules.append(new_rule)
            
            # Save to file
            self._save_rules()
            return True
            
        except Exception as e:
            logger.error(f"Error adding rule: {e}")
            return False
    
    def _save_rules(self) -> None:
        """Save current rules to file"""
        try:
            rules_data = {
                "rules": [
                    {
                        "id": rule.id,
                        "title": rule.title,
                        "description": rule.description,
                        "content": rule.content
                    }
                    for rule in self.rules
                ]
            }
            
            with open(self.rules_file, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving rules: {e}")
            raise
