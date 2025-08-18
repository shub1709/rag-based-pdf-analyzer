import requests
import json
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Represents an LLM response"""
    content: str
    model: str
    tokens_used: Optional[int] = None
    
class LLMManager:
    """Manages interaction with Ollama LLM"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get('llm', {}).get('base_url', 'http://localhost:11434')
        self.model_name = config.get('llm', {}).get('model_name', 'qwen3')
        self.temperature = config.get('llm', {}).get('temperature', 0.3)
        self.max_tokens = config.get('llm', {}).get('max_tokens', 2000)
        self.timeout = config.get('llm', {}).get('timeout', 1200)
        
        # Check connection
        self._check_ollama_connection()
    
    def _check_ollama_connection(self) -> None:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Successfully connected to Ollama")
                models = response.json().get('models', [])
                logger.info(f"Available models: {[m['name'] for m in models]}")
            else:
                logger.warning(f"Ollama connection returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {str(e)}")
            logger.warning("Make sure Ollama is running (ollama serve)")
    
    def generate(self, prompt: str, context: Optional[str] = None) -> LLMResponse:
        """
        Generate response from LLM
        
        Args:
            prompt: The prompt to send to the LLM
            context: Optional context to include
            
        Returns:
            LLMResponse object
        """
        try:
            # Prepare the request
            full_prompt = prompt
            if context:
                full_prompt = f"Context:\n{context}\n\nQuery:\n{prompt}"
            
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "temperature": self.temperature,
                "stream": False
            }
            
            # Make request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return LLMResponse(
                    content=result.get('response', ''),
                    model=self.model_name,
                    tokens_used=result.get('total_duration')
                )
            else:
                logger.error(f"LLM request failed with status {response.status_code}")
                return LLMResponse(
                    content="Error generating response. Please check the LLM connection.",
                    model=self.model_name
                )
        
        except requests.exceptions.Timeout:
            logger.error("LLM request timed out")
            return LLMResponse(
                content="Request timed out. The model might be processing a large request.",
                model=self.model_name
            )
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return LLMResponse(
                content=f"Error: {str(e)}",
                model=self.model_name
            )
    
    def summarize(self, text: str, summary_type: str = "medium") -> str:
        """
        Generate a summary of the given text
        
        Args:
            text: Text to summarize
            summary_type: Type of summary (short, medium, long)
            
        Returns:
            Summary text
        """
        length_instructions = {
            "short": "Provide a brief 2-3 sentence summary",
            "medium": "Provide a comprehensive summary in about 1 paragraph",
            "long": "Provide a detailed summary covering all major points"
        }
        
        prompt = f"""
        {length_instructions.get(summary_type, length_instructions['medium'])} of the following text:
        
        {text}
        
        Summary:
        """
        
        response = self.generate(prompt)
        return response.content
    
    def answer_question(self, question: str, context: str) -> str:
        """
        Answer a question based on provided context
        
        Args:
            question: The question to answer
            context: Relevant context from the document
            
        Returns:
            Answer to the question
        """
        prompt = f"""
        Based on the following context from a document, please answer the question.
        If the answer cannot be found in the context, say "I cannot find this information in the provided context."
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        response = self.generate(prompt)
        return response.content
    
    def extract_key_insights(self, text: str, num_insights: int = 5) -> List[str]:
        """
        Extract key insights from text
        
        Args:
            text: Text to analyze
            num_insights: Number of insights to extract
            
        Returns:
            List of key insights
        """
        prompt = f"""
        Extract {num_insights} key insights from the following text.
        Format each insight as a clear, concise bullet point.
        
        Text:
        {text}
        
        Key Insights:
        """
        
        response = self.generate(prompt)
        
        # Parse the response to extract insights
        insights = []
        for line in response.content.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                insights.append(line.lstrip('-•* '))
        
        return insights[:num_insights] if insights else [response.content]