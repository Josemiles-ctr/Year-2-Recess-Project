import json
import logging
from datetime import datetime, timezone

from flask import (
    Blueprint, request, jsonify, render_template, session, redirect, url_for, current_app
)
from flask_login import login_user, logout_user, login_required, current_user

from src.infrastructure.database import db, User, ChatSession, ChatMessage
from src.domain.entities import ChatMessage as ChatMessageEntity

logger = logging.getLogger(__name__)
web_bp = Blueprint("web", __name__)


# ── Auth ──────────────────────────────────────────────────────────────

@web_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("web.upload_page"))
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            next_page = session.pop("next", None)
            return redirect(next_page or url_for("web.upload_page"))
        error = "Invalid email or password."
    return render_template("login.html", error=error)


@web_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("web.upload_page"))
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        if not email or not password:
            error = "Email and password are required."
        elif password != confirm:
            error = "Passwords do not match."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif User.query.filter_by(email=email).first():
            error = "An account with this email already exists."
        else:
            user = User(email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            return redirect(url_for("web.upload_page"))
    return render_template("register.html", error=error)


@web_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("web.login"))


# ── Pages ─────────────────────────────────────────────────────────────

@web_bp.route("/")
def index():
    return render_template("index.html")


@web_bp.route("/upload")
@login_required
def upload_page():
    return render_template("upload.html")


@web_bp.route("/report")
@login_required
def report_page():
    sid = request.args.get("sid")
    if sid:
        session_obj = ChatSession.query.filter_by(id=sid, user_id=current_user.id).first()
        if not session_obj:
            return redirect(url_for("web.upload_page"))
    else:
        session_obj = (
            ChatSession.query.filter_by(user_id=current_user.id)
            .order_by(ChatSession.updated_at.desc())
            .first()
        )
        if not session_obj:
            return redirect(url_for("web.upload_page"))

    scan_data = json.loads(session_obj.scan_data or "{}")
    traditional_probs = sorted(scan_data.get("traditional_per_class_probabilities", {}).items(), key=lambda x: -x[1])
    cnn_probs = sorted(scan_data.get("cnn_per_class_probabilities", {}).items(), key=lambda x: -x[1])

    session_data = {
        "scan_filename": session_obj.scan_filename,
        "traditional_prediction": scan_data.get("traditional_prediction", "N/A"),
        "traditional_confidence": scan_data.get("traditional_confidence", 0),
        "traditional_probs": traditional_probs,
        "cnn_prediction": scan_data.get("cnn_prediction", "N/A"),
        "cnn_confidence": scan_data.get("cnn_confidence", 0),
        "cnn_probs": cnn_probs,
        "llm_narrative": scan_data.get("llm_narrative", ""),
    }

    sessions_list = (
        ChatSession.query.filter_by(user_id=current_user.id)
        .order_by(ChatSession.updated_at.desc())
        .all()
    )

    return render_template(
        "report.html",
        session_data=session_data,
        sessions=sessions_list,
        current_session_id=session_obj.id,
    )


# ── Session API ───────────────────────────────────────────────────────

@web_bp.route("/api/sessions", methods=["GET"])
@login_required
def list_sessions():
    sessions_list = (
        ChatSession.query.filter_by(user_id=current_user.id)
        .order_by(ChatSession.updated_at.desc())
        .all()
    )
    return jsonify({
        "status": "success",
        "sessions": [
            {"id": s.id, "title": s.title, "scan_filename": s.scan_filename,
             "created_at": s.created_at.isoformat(), "message_count": s.messages.count()}
            for s in sessions_list
        ],
    })


@web_bp.route("/api/sessions/<int:sid>", methods=["GET"])
@login_required
def get_session(sid):
    session_obj = ChatSession.query.filter_by(id=sid, user_id=current_user.id).first()
    if not session_obj:
        return jsonify({"status": "error", "message": "Session not found"}), 404
    messages = [
        {"role": m.role, "content": m.content}
        for m in session_obj.messages.order_by(ChatMessage.created_at).all()
    ]
    scan_data = json.loads(session_obj.scan_data or "{}")
    return jsonify({
        "status": "success",
        "session": {
            "id": session_obj.id,
            "title": session_obj.title,
            "scan_filename": session_obj.scan_filename,
            "history": messages,
            "scan_data": scan_data,
        },
    })


@web_bp.route("/api/sessions/<int:sid>", methods=["DELETE"])
@login_required
def delete_session(sid):
    session_obj = ChatSession.query.filter_by(id=sid, user_id=current_user.id).first()
    if session_obj:
        db.session.delete(session_obj)
        db.session.commit()
    return jsonify({"status": "success"})


@web_bp.route("/api/sessions/<int:sid>/title", methods=["PUT"])
@login_required
def update_session_title(sid):
    session_obj = ChatSession.query.filter_by(id=sid, user_id=current_user.id).first()
    if not session_obj:
        return jsonify({"status": "error", "message": "Session not found"}), 404
    data = request.get_json() or {}
    title = data.get("title", "").strip()
    if title:
        session_obj.title = title
        db.session.commit()
    return jsonify({"status": "success"})


# ── Analyze ───────────────────────────────────────────────────────────

@web_bp.route("/api/analyze", methods=["POST"])
@login_required
def analyze_scan():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No file selected."}), 400

    try:
        controller = current_app.config["ANALYZE_CONTROLLER"]
        response_data = controller.handle_upload(file.filename, file.read())

        if response_data["status"] == "success":
            scan_data = {
                "traditional_prediction": response_data["traditional_model"]["prediction"],
                "traditional_confidence": response_data["traditional_model"]["confidence"],
                "traditional_per_class_probabilities": response_data["traditional_model"].get("per_class_probabilities", {}),
                "cnn_prediction": response_data["cnn_model"]["prediction"],
                "cnn_confidence": response_data["cnn_model"]["confidence"],
                "cnn_per_class_probabilities": response_data["cnn_model"].get("per_class_probabilities", {}),
                "llm_narrative": response_data.get("llm_narrative", ""),
            }
            session_obj = ChatSession(
                user_id=current_user.id,
                title=response_data["traditional_model"]["prediction"],
                scan_filename=file.filename,
                scan_data=json.dumps(scan_data),
            )
            db.session.add(session_obj)
            db.session.commit()
            response_data["session_id"] = session_obj.id

        return jsonify(response_data)
    except Exception as e:
        logger.exception("Analyze failed")
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Chat ──────────────────────────────────────────────────────────────

@web_bp.route("/api/chat", methods=["POST"])
@login_required
def chat_assistant():
    data = request.get_json() or {}
    message = data.get("message")
    sid = data.get("session_id")
    if not message:
        return jsonify({"status": "error", "message": "Empty query."}), 400
    if not sid:
        return jsonify({"status": "error", "message": "session_id required."}), 400

    session_obj = ChatSession.query.filter_by(id=sid, user_id=current_user.id).first()
    if not session_obj:
        return jsonify({"status": "error", "message": "Session not found."}), 404

    scan_data = json.loads(session_obj.scan_data or "{}")
    diagnostic_context = {
        "traditional_prediction": scan_data.get("traditional_prediction", ""),
        "traditional_confidence": scan_data.get("traditional_confidence", 0),
        "traditional_per_class_probabilities": scan_data.get("traditional_per_class_probabilities", {}),
        "cnn_prediction": scan_data.get("cnn_prediction", ""),
        "cnn_confidence": scan_data.get("cnn_confidence", 0),
        "cnn_per_class_probabilities": scan_data.get("cnn_per_class_probabilities", {}),
        "llm_narrative": scan_data.get("llm_narrative", ""),
    }

    try:
        controller = current_app.config["CHAT_CONTROLLER"]
        db_history = (
            session_obj.messages.order_by(ChatMessage.created_at).all()
        )
        history = [
            ChatMessageEntity(role=m.role, content=m.content) for m in db_history
        ]

        result = controller.handle_message(
            [{"role": m.role, "content": m.content} for m in db_history],
            message,
            diagnostic_context,
        )

        if result["status"] == "success":
            user_msg = ChatMessage(session_id=session_obj.id, role="user", content=message)
            assistant_msg = ChatMessage(
                session_id=session_obj.id,
                role="assistant",
                content=result["response"]["content"],
            )
            db.session.add_all([user_msg, assistant_msg])
            session_obj.updated_at = datetime.now(timezone.utc)
            db.session.commit()

        return jsonify(result)
    except Exception as e:
        logger.exception("Chat failed")
        return jsonify({"status": "error", "message": str(e)}), 500


@web_bp.route("/api/chat/clear", methods=["POST"])
@login_required
def clear_chat():
    sid = (request.get_json() or {}).get("session_id")
    if sid:
        session_obj = ChatSession.query.filter_by(id=sid, user_id=current_user.id).first()
        if session_obj:
            ChatMessage.query.filter_by(session_id=session_obj.id).delete()
            db.session.commit()
    return jsonify({"status": "success", "message": "Chat cleared."})