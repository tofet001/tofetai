@app.route('/chatbot', methods=['POST'])
@login_required
def chat():
    user_input = request.json.get('message')
    user_id = current_user.id

    # Save user message to database
    save_chat_history(user_id, user_input, 'user')

    # Get bot response
    bot_response = asyncio.run(chatbot.handle_task(user_input))

    # Save bot response to database
    save_chat_history(user_id, bot_response, 'bot')

    return jsonify({'response': bot_response})

def save_chat_history(user_id, message, message_type):
    # Implement database insertion logic
    pass