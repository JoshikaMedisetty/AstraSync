// frontend/js/api.js

// Backend running on your laptop
const API_BASE = "http://127.0.0.1:5001";

async function apiPost(path, payload) {
  const res = await fetch(API_BASE + path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload),
  });

  const text = await res.text();

  if (!res.ok) {
    throw new Error(text || `HTTP ${res.status}`);
  }

  return text ? JSON.parse(text) : {};
}

async function apiGet(path) {
  const res = await fetch(API_BASE + path);

  const text = await res.text();

  if (!res.ok) {
    throw new Error(text || `HTTP ${res.status}`);
  }

  return text ? JSON.parse(text) : {};
}
