import os
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st

SPAM_ENDPOINT = os.getenv("SPAM_ENDPOINT")
SENTIMENT_ENDPOINT = os.getenv("SENTIMENT_ENDPOINT") or None  # Not implemented yet


def call_model(endpoint: str, text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        payload = {"text": text}
        resp = requests.post(endpoint, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, str(e)


def extract_label_and_proba(result: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    """
    Try to infer a spam label and probability from various possible API response formats.
    This is defensive: it will work even if the backend returns slightly different keys.
    """
    # --- Label extraction ---
    label = result.get("label") or result.get("prediction") or result.get("class")
    # Some APIs return lists
    if isinstance(label, list) and label:
        label = label[0]
    if label is not None:
        label = str(label)

    # --- Probability extraction ---
    proba = None
    proba = result["probability"] if "probability" in result else proba

    if isinstance(proba, dict) and proba:
        # Prefer explicit "spam" key if present
        if "spam" in proba:
            proba = proba["spam"]
        else:
            # Fallback: take the highest probability
            proba = max(proba.values())

    # If probability is a list
    if isinstance(proba, list) and proba:
        proba = proba[0]

    # Ensure it's a float between 0 and 1 if possible
    if isinstance(proba, (int, float)):
        # If it looks like a percentage ( > 1 ), normalize
        if proba > 1.0:
            proba = proba / 100.0
    else:
        proba = None

    # If we have a probability but no label, infer label: >= 0.5 -> spam
    if proba is not None and label is None:
        label = "spam" if proba >= 0.5 else "ham"

    return label, proba


def display_spam_result(result: Dict[str, Any]) -> None:
    label, proba = extract_label_and_proba(result)

    # Default text if we can't infer anything
    if label is None and proba is None:
        st.warning("The API returned a response, but I couldn't interpret it.")
        st.json(result)
        return

    # Normalize label
    norm_label = label.lower() if isinstance(label, str) else None
    # Prioritize probability: > 0.5 => spam, otherwise fall back to label
    if proba is not None:
        is_spam = proba >= 0.5
    else:
        is_spam = norm_label in {"spam", "1", "true"}

    # Build a nice message
    if proba is not None:
        pct = proba * 100
        if is_spam:
            st.error(f"🚨 Result: **SPAM** ({pct:.1f}% confidence)")
        else:
            st.success(f"✅ Result: **NOT SPAM** ({pct:.1f}% confidence)")
        st.progress(min(max(proba, 0.0), 1.0))
    else:
        # No probability, only label
        if is_spam:
            st.error("🚨 Result: **SPAM**")
        else:
            st.success("✅ Result: **NOT SPAM**")

    # Raw JSON for debugging / inspection
    with st.expander("Raw API response"):
        st.json(result)


def main():
    st.set_page_config(page_title="Model UI - Spam & Sentiment", page_icon="🤖")
    st.title("🤖 Unified Model UI")
    st.write(
        "This interface lets you test two models:\n"
        "- a **spam detector** (available now)\n"
        "- a **sentiment classifier** (coming soon)"
    )

    user_text = st.text_area("📝 Enter some text:", height=150)

    col1, col2 = st.columns(2)
    with col1:
        spam_clicked = st.button("🔍 Is it spam?")
    with col2:
        sentiment_clicked = st.button("🎭 Sentiment analysis")

    if not (spam_clicked or sentiment_clicked):
        return

    if not user_text.strip():
        st.warning("Please enter some text first.")
        return

    if spam_clicked:
        st.subheader("Spam model result")
        result, error = call_model(SPAM_ENDPOINT, user_text)
        if error:
            st.error(error)
        elif result is None:
            st.error("Empty response from the API.")
        else:
            display_spam_result(result)

    if sentiment_clicked:
        st.subheader("Sentiment model result")
        if SENTIMENT_ENDPOINT is None:
            st.info("🚧 The sentiment service is not available yet (no endpoint configured).")
        else:
            result, error = call_model(SENTIMENT_ENDPOINT, user_text)
            st.error(error) if error else st.json(result)


if __name__ == "__main__":
    main()
