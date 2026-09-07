"""
AI Brain Module
Handles the intelligence and conversation logic using OpenAI GPT
"""

import openai
import os
from dotenv import load_dotenv

load_dotenv()

class AIBrain:
    def __init__(self, api_key, model="gpt-4", temperature=0.7):
        """
        Initialize AI Brain
        
        Args:
            api_key (str): OpenAI API key
            model (str): GPT model to use
            temperature (float): Creativity level (0-1)
        """
        openai.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.system_prompt = """You are JARVIS, an intelligent voice assistant inspired by the AI from Iron Man. 
        You are helpful, witty, and professional. Keep responses concise but informative.
        When the user asks you to do something, you should acknowledge and help them.
        Always maintain a respectful and intelligent demeanor."""
    
    def get_response(self, conversation_history):
        """
        Get AI response based on conversation history
        
        Args:
            conversation_history (list): List of message dicts with 'role' and 'content'
            
        Returns:
            str: AI response
        """
        try:
            # Prepare messages with system prompt
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(conversation_history)
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=500
            )
            
            # Extract response text
            return response['choices'][0]['message']['content'].strip()
        
        except Exception as e:
            print(f"❌ Error getting AI response: {e}")
            return "I apologize, but I encountered an error processing your request."
    
    def set_system_prompt(self, prompt):
        """
        Customize the system prompt
        
        Args:
            prompt (str): New system prompt
        """
        self.system_prompt = prompt
        print("✓ System prompt updated")
    
    def get_system_prompt(self):
        """Get current system prompt"""
        return self.system_prompt
