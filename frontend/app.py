"""
Streamlit UI for SHL Assessment Recommender — Markdown Trace View.
Connects to the FastAPI backend via HTTP.
"""
import os
import streamlit as st
import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page Setup ──────────────────────────────────────────────────

st.set_page_config(
    page_title="SHL Markdown Trace Viewer",
    page_icon="📝",
    layout="wide",
)

st.title("📝 SHL Markdown Trace Viewer")
st.caption("Displays the conversation matching the evaluation trace format.")

# ── Sidebar ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("About")
    st.markdown("This view renders the conversation strictly in the trace markdown format.")
    st.divider()

    # Health check
    try:
        r = httpx.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200:
            data = r.json()
            st.success(f"✅ Backend connected")
        else:
            st.error("⚠️ Backend unhealthy")
    except Exception:
        st.error("❌ Cannot reach backend")

    if st.button("Clear Conversation History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Chat State ──────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Helper to render Markdown ───────────────────────────────────

def render_markdown_trace(messages):
    """Renders the entire conversation history in trace format."""
    st.markdown("## Conversation")
    
    turn = 1
    for i in range(0, len(messages), 2):
        st.markdown(f"### Turn {turn}")
        
        user_msg = messages[i]
        st.markdown(f"> {user_msg['content']}")
        st.markdown("")
        
        if i + 1 < len(messages):
            st.markdown("**Agent**")
            st.markdown("")
            
            agent_msg = messages[i+1]
            st.markdown(agent_msg['content'])
            st.markdown("")
            
            recs = agent_msg.get('recommendations', [])
            if recs:
                table = "| # | Name | Test Type | Keys | Duration | Languages | URL |\n"
                table += "|---|------|-----------|------|----------|-----------|-----|\n"
                for idx, r in enumerate(recs, 1):
                    # Format URL as <url> per trace
                    url = f"<{r.get('url', '')}>"
                    table += f"| {idx} | {r.get('name')} | {r.get('test_type')} | {r.get('keys')} | {r.get('duration')} | {r.get('languages')} | {url} |\n"
                st.markdown(table)
            else:
                st.markdown("_No recommendations this turn (`recommendations: null`)._")
            
            st.markdown("")
            eoc = str(agent_msg.get('end_of_conversation', False)).lower()
            st.markdown(f"_`end_of_conversation`: **{eoc}**_")
            st.markdown("")
            
        turn += 1

# ── Display Chat History ────────────────────────────────────────

# We render the full history statically
if st.session_state.messages:
    render_markdown_trace(st.session_state.messages)

# ── Handle User Input ───────────────────────────────────────────

prompt = st.chat_input("Enter a message...")

if prompt:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Build payload
    payload = {
        "messages": [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
    }
    
    with st.spinner("Waiting for API response..."):
        try:
            r = httpx.post(
                f"{API_URL}/chat",
                json=payload,
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            
            # Append assistant's response
            st.session_state.messages.append({
                "role": "assistant",
                "content": data.get("reply", "No response"),
                "recommendations": data.get("recommendations", []),
                "end_of_conversation": data.get("end_of_conversation", False)
            })
            st.rerun()
            
        except httpx.HTTPStatusError as e:
            st.error(f"API error: {e.response.status_code}")
            st.session_state.messages.pop()
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.messages.pop()
