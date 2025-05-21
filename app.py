from flask import Flask, request, jsonify
import asyncio

app = Flask(__name__)

# Initialize the chatbot
chatbot = TofetAI(main.py)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'response': 'Please provide a message.'}), 400

    # Process the user input and get the response from the chatbot
    response = asyncio.run(chatbot.handle_task(user_input))
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)