import datetime
from typing import Dict, List, Optional


class SessionStore:
    def __init__(self):
        self._data: Dict[str, Dict[str, dict]] = {}

    def _user_key(self, user: str) -> str:
        return user.lower().strip()

    def list_sessions(self, user: str) -> List[dict]:
        key = self._user_key(user)
        sessions = self._data.get(key, {})
        result = []
        for sid, s in sessions.items():
            result.append({
                "id": sid,
                "title": s.get("title", "New chat"),
                "created_at": s.get("created_at", ""),
                "message_count": len(s.get("history", [])),
            })
        result.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return result

    def create_session(self, user: str, sid: str):
        key = self._user_key(user)
        if key not in self._data:
            self._data[key] = {}
        self._data[key][sid] = {
            "title": "New chat",
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "history": [],
        }

    def get_session(self, user: str, sid: str) -> Optional[dict]:
        key = self._user_key(user)
        s = self._data.get(key, {}).get(sid)
        if s is None:
            return None
        return {
            "id": sid,
            "title": s["title"],
            "created_at": s["created_at"],
            "history": s["history"],
        }

    def get_history(self, user: str, sid: str) -> List[dict]:
        key = self._user_key(user)
        s = self._data.get(key, {}).get(sid)
        if s is None:
            return []
        return s["history"]

    def append_message(self, user: str, sid: str, msg: dict):
        key = self._user_key(user)
        s = self._data.get(key, {}).get(sid)
        if s is not None:
            s["history"].append(msg)

    def update_title(self, user: str, sid: str, title: str):
        key = self._user_key(user)
        s = self._data.get(key, {}).get(sid)
        if s is not None:
            s["title"] = title

    def delete_session(self, user: str, sid: str):
        key = self._user_key(user)
        self._data.get(key, {}).pop(sid, None)
