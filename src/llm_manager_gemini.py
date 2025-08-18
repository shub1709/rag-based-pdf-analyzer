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

    def rewrite_query(self, query: str) -> str:
        """Rewrite user query for clarity and retrieval effectiveness"""
        prompt = f"""
        You are optimizing a user question for document retrieval.

        Rewrite the question into ONE detailed, standalone query that:
        - preserves meaning
        - removes pronouns/ambiguity (generalize unclear referents, e.g., "the individual described in the document")
        - adds likely domain synonyms in parentheses (e.g., deductible (excess, out-of-pocket))
        - includes relevant constraints (dates, sections, document type) if implied
        - avoids meta text, instructions, options, lists, quotes, or asking for clarification

        Examples:
        Q: What's the deductible?
        A: Identify the deductible (excess, out-of-pocket amount) defined in the insurance policy document, including the amount, conditions, and when it applies.

        Q: tell me about his work experience and the tools he has used
        A: Provide a detailed summary of the individual described in the document, including roles, employers, dates, responsibilities, and the key tools/technologies (software, frameworks, platforms) used in each role.

        Now rewrite this:

        Original Question: {query}
        Rewritten Question:
        """
        response = self.generate(prompt)
        rewritten = response.content.strip() if response and response.content else query
        self.last_rewrite = rewritten  # for Streamlit UI debugging
        return rewritten

    def expand_query(self, query: str, num_variants: int = 4) -> List[str]:
        """Generate multiple alternative phrasings of the query"""
        prompt = f"""
        Create {num_variants} diverse retrieval queries for the following question.
        Mix styles among:
        1) Verbose NL (rich detail, synonyms in parentheses)
        2) Keyword-only (comma-separated terms & synonyms)
        3) Boolean-style (use OR for synonyms, minimal punctuation)
        4) Concise focused question
        Rules:
        - Preserve meaning
        - No clarifying questions, no placeholders, no lists, no quotes, no explanations
        - Output EXACTLY {num_variants} lines, one query per line

        Question: {query}

        Variants:
        """
        response = self.generate(prompt)

        variants = []
        if response and response.content:
            for line in response.content.split("\n"):
                line = line.strip(" -•*")
                if line:
                    variants.append(line)
        
        self.last_expansions = variants[:num_variants]

        # Fallback: at least return original query
        if not variants:
            variants = [query]

        return self.last_expansions

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
