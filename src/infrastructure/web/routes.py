from flask import Blueprint, request, jsonify, render_template, session, redirect, url_for, current_app

web_bp = Blueprint('web', __name__)

@web_bp.route('/')
def index():
    """Task Assignee Implementation steps:
    1. Render and return the home landing page template (index.html).
    """
    # Placeholder: Renders the home screen
    return render_template('index.html')


@web_bp.route('/upload')
def upload_page():
    """Render the dedicated scan-upload workspace."""
    return render_template('upload.html')

@web_bp.route('/report')
def report_page():
    """Task Assignee Implementation steps:
    1. Check if 'active_prediction' details exist inside the cookie session.
    2. If missing, redirect the browser to the upload landing page.
    3. If present, render the diagnostics dashboard template (report.html), passing session details.
    """
    report_data = session.get('active_prediction')
    if not report_data:
        return redirect(url_for('web.upload_page'))

    return render_template('report.html', data=report_data)

@web_bp.route('/api/analyze', methods=['POST'])
def analyze_scan():
    """Task Assignee Implementation steps:
    1. Check if 'file' exists in the request.files dictionary. If not, return a 400 Bad Request error.
    2. Parse the uploaded file stream and extract filename and raw file bytes.
    3. Fetch the 'ANALYZE_CONTROLLER' instance from the current_app.config environment.
    4. Call controller.handle_upload(filename, file_bytes).
    5. If prediction returns success:
       - Save the resulting dictionary to session['active_prediction'].
       - Reset the conversational log by clearing session['chat_history'].
    6. Return the dictionary back to the client as a JSON payload.
    """
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded."}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected."}), 400

    # Placeholder: Invoke the injected controller to process the upload
    try:
        controller = current_app.config['ANALYZE_CONTROLLER']
        response_data = controller.handle_upload(file.filename, file.read())
        
        if response_data["status"] == "success":
            session['active_prediction'] = response_data
            session['chat_history'] = []
            
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Routing/Controller error: {str(e)}"}), 500

@web_bp.route('/api/chat', methods=['POST'])
def chat_assistant():
    """Task Assignee Implementation steps:
    1. Parse user queries from request.get_json().
    2. Extract 'active_prediction' from session. If missing, return an error.
    3. Retrieve the current chat log list from session['chat_history'].
    4. Construct a diagnostic context dict containing verdict details.
    5. Fetch 'CHAT_CONTROLLER' from current_app.config.
    6. Call controller.handle_message(chat_history, new_message, diagnostic_context).
    7. If chat execution succeeds:
       - Append user message and chatbot response to chat_history list.
       - Save the updated history list back to session['chat_history'].
    8. Return response payload as JSON.
    """
    data = request.get_json() or {}
    message = data.get('message')
    if not message:
        return jsonify({"status": "error", "message": "Empty query."}), 400

    report_data = session.get('active_prediction')
    if not report_data:
        return jsonify({"status": "error", "message": "No active scan context."}), 400

    # Placeholder: Invoke the injected controller to execute follow-up dialogues
    try:
        controller = current_app.config['CHAT_CONTROLLER']
        chat_history = session.get('chat_history', [])
        
        diagnostic_context = {
            "verdict": report_data["consensus"]["verdict"],
            "risk_level": report_data["consensus"]["risk_level"],
            "traditional_prediction": report_data["traditional_model"]["prediction"],
            "traditional_confidence": report_data["traditional_model"]["confidence"],
            "cnn_prediction": report_data["cnn_model"]["prediction"],
            "cnn_confidence": report_data["cnn_model"]["confidence"]
        }

        result = controller.handle_message(chat_history, message, diagnostic_context)
        
        if result["status"] == "success":
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": result["response"]["content"]})
            session['chat_history'] = chat_history
            
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Routing/Controller error: {str(e)}"}), 500

@web_bp.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """Task Assignee Implementation steps:
    1. Clear or overwrite session['chat_history'] to empty list.
    2. Return success response status.
    """
    # Placeholder: Empty the chat array
    session['chat_history'] = []
    return jsonify({"status": "success", "message": "Chat cleared."})
