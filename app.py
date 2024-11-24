# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from typing import List
import openai

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuración de producción
PORT = int(os.getenv('PORT', 10000))

class LlamaChat:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('5ddf75fe-075b-4f5c-9e9b-a92cb579faf3')
        
        if not self.api_key:
            raise ValueError("API key must be provided")
            
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.sambanova.ai/v1"
        )
        
        self.conversation_history: List[dict] = [
            {"role": "system", "content": "You are a helpful assistant"}
        ]

    def send_message(self, message: str, temperature: float = 0.1, top_p: float = 0.1) -> str:
        self.conversation_history.append({"role": "user", "content": message})
        try:
            response = self.client.chat.completions.create(
                model='Meta-Llama-3.1-8B-Instruct',
                messages=self.conversation_history,
                temperature=temperature,
                top_p=top_p
            )
            assistant_message = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            return assistant_message
        except Exception as e:
            return f"Error: {str(e)}"

    def get_conversation_history(self) -> List[dict]:
        return self.conversation_history

    def clear_conversation(self) -> None:
        self.conversation_history = [
            {"role": "system", "content": "You are a helpful assistant"}
        ]

# Crear instancia global de LlamaChat
chat = LlamaChat()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "Welcome to LlamaChat API",
        "endpoints": {
            "/chat": "POST - Send a message to the chat",
            "/history": "GET - Get conversation history",
            "/clear": "POST - Clear conversation history"
        }
    })

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        message = data.get('message')
        temperature = data.get('temperature', 0.1)
        top_p = data.get('top_p', 0.1)
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
            
        response = chat.send_message(message, temperature, top_p)
        return jsonify({
            'response': response,
            'history': chat.get_conversation_history()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    try:
        return jsonify({'history': chat.get_conversation_history()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_history():
    try:
        chat.clear_conversation()
        return jsonify({'message': 'Conversation history cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)