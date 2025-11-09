"""
Model interfaces for different LLM providers
"""

import os
import subprocess
import json
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseModelInterface(ABC):
    """Base interface for all model implementations"""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response to the prompt"""
        pass

class OllamaInterface(BaseModelInterface):
    """Interface for Ollama models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Check if model is available
        self._check_model()
    
    def _check_model(self):
        """Check if model is pulled, pull if not"""
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        if self.model_name not in result.stdout:
            print(f"Pulling model {self.model_name}...")
            subprocess.run(["ollama", "pull", self.model_name])
    
    def generate(self, prompt: str) -> str:
        """Generate response using Ollama"""
        result = subprocess.run(
            ["ollama", "run", self.model_name, prompt],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()

class HuggingFaceInterface(BaseModelInterface):
    """Interface for HuggingFace models"""
    
    def __init__(self, model_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    
    def generate(self, prompt: str) -> str:
        """Generate response using HuggingFace model"""
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        response = response[len(prompt):]
        return response.strip()

class GeminiInterface(BaseModelInterface):
    """Interface for Google Gemini models"""

    def __init__(self, model_name: str = "gemini-pro"):
        import google.generativeai as genai

        self.model_name = model_name
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        """Generate response using Gemini API"""
        # Add system instruction as part of the prompt
        full_prompt = "You are a helpful assistant that solves physics problems.\n\n" + prompt

        response = self.model.generate_content(
            full_prompt,
            generation_config={
                'temperature': 0.7,
                'max_output_tokens': 1000,
            }
        )
        return response.text.strip()

class AnthropicInterface(BaseModelInterface):
    """Interface for Anthropic Claude models"""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229"):
        import anthropic
        
        self.model_name = model_name
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate(self, prompt: str) -> str:
        """Generate response using Anthropic API"""
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()

def get_model_interface(model_name: str) -> BaseModelInterface:
    """Factory function to get the appropriate model interface"""

    # Ollama models (contain ':')
    if ':' in model_name:
        return OllamaInterface(model_name)

    # Gemini models
    elif model_name.startswith('gemini'):
        return GeminiInterface(model_name)

    # Claude models
    elif model_name.startswith('claude'):
        return AnthropicInterface(model_name)

    # Default to HuggingFace
    else:
        return HuggingFaceInterface(model_name)
