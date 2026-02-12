import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from services.db_service import get_conn, init_db, save_profile, get_profile, save_log, get_logs
from ml.dual_baseline import score_entry

from services.scoring_engine import score_all
from services.google_fit_service import (
    get_authorization_url,
    exchange_code_for_tokens,
    refresh_access_token,
    fetch_aggregated_data
)

from services.db_service import save_tokens, get_tokens
import time


load_dotenv()

app = Flask(__name__)
CORS(app)

conn = get_conn()
init_db(conn)

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/profile")
def set_profile():
    data = request.get_json(force=True)
    user_id = data.get("user_id", "demo_user")
    save_profile(conn, user_id, data)
    return jsonify({"ok": True})

@app.post("/submit_data")
def submit_data():
    entry = request.get_json(force=True)
    user_id = entry.get("user_id", "demo_user")
    date = entry.get("date")
    if not date:
        return jsonify({"ok": False, "error": "date required"}), 400
    save_log(conn, user_id, date, entry)
    return jsonify({"ok": True})

@app.post("/score")
def score():
    entry = request.get_json(force=True)
    user_id = entry.get("user_id", "demo_user")

    profile = get_profile(conn, user_id)
    logs = get_logs(conn, user_id, limit=14)

    result = score_entry(profile, entry, logs)
    return jsonify(result)

@app.route("/")
def home():
    return {"ok": True, "message": "AstraSync backend running", "try": ["/health", "/score"]}, 200

@app.get("/history/<user_id>")
def history(user_id):
    logs = get_logs(conn, user_id, limit=60)
    return jsonify({"logs": logs})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

@app.route("/auth/google-fit")
def auth_google_fit():
    url = get_authorization_url()
    return redirect(url)
@app.route("/auth/google/callback")
def google_callback():
    code = request.args.get("code")

    tokens = exchange_code_for_tokens(code)

    if "access_token" not in tokens:
        return jsonify({"error": "Token exchange failed"}), 400

    conn = get_db_connection()

    save_tokens(
        conn,
        user_id="demo_user",   # replace with real logged-in user later
        provider="google_fit",
        access_token=tokens["access_token"],
        refresh_token=tokens.get("refresh_token"),
        expires_in=tokens["expires_in"]
    )

    return jsonify({"message": "Google Fit connected successfully"})

@app.route("/sync/google-fit", methods=["POST"])
def sync_google_fit():

    conn = get_db_connection()
    token_row = get_tokens(conn, "demo_user", "google_fit")

    if not token_row:
        return jsonify({"error": "Google Fit not connected"}), 400

    access_token, refresh_token, expires_at = token_row

    # Auto refresh if expired
    if time.time() > expires_at:
        new_tokens = refresh_access_token(refresh_token)

        access_token = new_tokens["access_token"]

        save_tokens(
            conn,
            "demo_user",
            "google_fit",
            access_token,
            refresh_token,
            new_tokens["expires_in"]
        )

    data = fetch_aggregated_data(access_token)

    return jsonify(data)

