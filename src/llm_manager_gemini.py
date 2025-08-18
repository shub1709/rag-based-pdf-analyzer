"""LLM interaction using Google Gemini"""

import google.generativeai as genai
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
    """Manages interaction with Gemini LLM"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('llm', {}).get('api_key')
        self.model_name = config.get('llm', {}).get('model_name', 'gemini-pro')
        self.temperature = config.get('llm', {}).get('temperature', 0.7)
        self.max_tokens = config.get('llm', {}).get('max_tokens', 2000)

        if not self.api_key:
            raise ValueError("Gemini API key not found in config.yaml")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def generate(self, prompt: str, context: Optional[str] = None) -> LLMResponse:
        try:
            full_prompt = f"Context:\n{context}\n\nQuery:\n{prompt}" if context else prompt
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            return LLMResponse(
                content=response.text,
                model=self.model_name
            )
        except Exception as e:
            logger.error(f"Gemini generation error: {str(e)}")
            return LLMResponse(content=f"Error: {str(e)}", model=self.model_name)

    def summarize(self, text: str, summary_type: str = "medium") -> str:
        length_instructions = {
            "short": "Provide a brief 2-3 sentence summary",
            "medium": "Provide a comprehensive summary in about 1 paragraph",
            "long": "Provide a detailed summary covering all major points"
        }
        prompt = f"""{length_instructions.get(summary_type)} of the following text:\n\n{text}"""
        return self.generate(prompt).content

    def answer_question(self, question: str, context: str) -> str:
        prompt = f"""
        Based on the following context from a document, answer the question.
        If the answer is not found, reply: "I cannot find this information in the provided context."

        Context:
        {context}

        Question: {question}
        """
        return self.generate(prompt).content

    def extract_key_insights(self, text: str, num_insights: int = 5) -> List[str]:
        prompt = f"""
        Extract {num_insights} key insights from the following text.
        Format each insight as a clear, concise bullet point.

        Text:
        {text}
        """
        response = self.generate(prompt)
        insights = []
        for line in response.content.split("\n"):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                insights.append(line.lstrip('-•* '))
        return insights[:num_insights] if insights else [response.content]
