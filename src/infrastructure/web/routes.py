import uuid
from flask import (
    Blueprint,
    request,
    jsonify,
    render_template,
    session,
    current_app,
    Response,
)

web_bp = Blueprint("web", __name__)


def _get_store():
    return current_app.config["SESSION_STORE"]


@web_bp.route("/")
def index():
    name = session.get("user_name", "")
    return render_template("index.html", user_name=name)


@web_bp.route("/api/set-name", methods=["POST"])
def set_name():
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    session["user_name"] = name or "Guest"
    return jsonify({"status": "success", "name": session["user_name"]})


@web_bp.route("/api/sessions", methods=["GET"])
def list_sessions():
    user = session.get("user_name", "Guest")
    store = _get_store()
    return jsonify({
        "status": "success",
        "sessions": store.list_sessions(user),
    })


@web_bp.route("/api/sessions", methods=["POST"])
def create_session():
    user = session.get("user_name", "Guest")
    store = _get_store()
    sid = str(uuid.uuid4())
    store.create_session(user, sid)
    return jsonify({"status": "success", "session_id": sid})


@web_bp.route("/api/sessions/<sid>", methods=["GET"])
def get_session(sid):
    user = session.get("user_name", "Guest")
    store = _get_store()
    data = store.get_session(user, sid)
    if data is None:
        return jsonify({"status": "error", "message": "Session not found"}), 404
    return jsonify({"status": "success", "session": data})


@web_bp.route("/api/sessions/<sid>", methods=["DELETE"])
def delete_session(sid):
    user = session.get("user_name", "Guest")
    store = _get_store()
    store.delete_session(user, sid)
    return jsonify({"status": "success"})


@web_bp.route("/api/sessions/<sid>/title", methods=["PUT"])
def update_session_title(sid):
    user = session.get("user_name", "Guest")
    data = request.get_json() or {}
    title = data.get("title", "").strip()
    store = _get_store()
    store.update_title(user, sid, title or "New chat")
    return jsonify({"status": "success"})


@web_bp.route("/api/chat", methods=["POST"])
def chat_assistant():
    data = request.get_json() or {}
    message = data.get("message")
    sid = data.get("session_id")
    if not message:
        return jsonify({"status": "error", "message": "Empty query."}), 400
    if not sid:
        return jsonify({"status": "error", "message": "session_id required."}), 400

    user = session.get("user_name", "Guest")
    store = _get_store()

    try:
        rag = current_app.config["RAG_SERVICE"]
        history = store.get_history(user, sid)

        from src.domain.entities import ChatMessage

        history_objs = [
            ChatMessage(role=item["role"], content=item["content"])
            for item in history
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]

        is_new = len(history) == 0
        result = rag.generate(message, history_objs)

        store.append_message(user, sid, {"role": "user", "content": message})
        store.append_message(user, sid, {"role": "assistant", "content": result})
        if is_new:
            title = rag.generate_title(message)
            store.update_title(user, sid, title)

        return jsonify({"status": "success", "response": {"content": result}})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@web_bp.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    data = request.get_json() or {}
    message = data.get("message")
    sid = data.get("session_id")
    if not message:
        return jsonify({"status": "error", "message": "Empty query."}), 400
    if not sid:
        return jsonify({"status": "error", "message": "session_id required."}), 400

    user = session.get("user_name", "Guest")
    store = _get_store()
    rag = current_app.config["RAG_SERVICE"]
    history = store.get_history(user, sid)

    from src.domain.entities import ChatMessage

    history_objs = [
        ChatMessage(role=item["role"], content=item["content"])
        for item in history
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ]

    def generate():
        full_text = ""
        for chunk in rag.generate_stream(message, history_objs):
            full_text += chunk
            yield f"data: {chunk}\n\n"

        store.append_message(user, sid, {"role": "user", "content": message})
        store.append_message(user, sid, {"role": "assistant", "content": full_text})
        if len(history) == 0:
            title = rag.generate_title(message)
            store.update_title(user, sid, title)
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")
