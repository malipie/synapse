import streamlit as st
import requests
import os
import time
import json

# --- Configuration ---
# Wewnątrz sieci Docker, frontend widzi backend pod nazwą serwisu 'synapse-backend'
API_URL = os.getenv("API_URL", "http://synapse-backend:8000/api/v1")

st.set_page_config(
    page_title="Synapse Enterprise",
    page_icon="🧠",
    layout="wide"
)

# --- Header ---
st.title("Synapse 🧠 ⟷ 🤖")
st.markdown("### Enterprise Medical RAG & Agentic Platform")
st.markdown("*(Powered by AutoGen, Arq & Qdrant)*")

# --- Sidebar: Ingest ---
with st.sidebar:
    st.header("🗂️ Document Ingestion")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file:
        if st.button("Ingest Document"):
            with st.spinner("Uploading & Processing (Fast Mode)..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    response = requests.post(f"{API_URL}/ingest", files=files)
                    
                    if response.status_code == 200:
                        st.success("Document processed & vectorized successfully!")
                        st.json(response.json())
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

    st.divider()
    st.info("System Status: Async Agents Ready ⚡")

# --- Chat Interface ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input & Async Workflow ---
if prompt := st.chat_input("Ask a question regarding the medical documents..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Assistant Logic (Async Polling)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # A. Enqueue Task (Fire & Forget)
            # Send the message + history context
            payload = {
                "message": prompt,
                "chat_history": [m for m in st.session_state.messages if m["role"] != "system"]
            }
            
            enqueue_resp = requests.post(f"{API_URL}/chat/enqueue", json=payload)
            
            if enqueue_resp.status_code == 202:
                task_data = enqueue_resp.json()
                task_id = task_data["task_id"]
                
                # B. Polling Loop (Waiting for Agents)
                # We use st.status to show a cool spinner with steps
                with st.status("🚀 Agents are working...", expanded=True) as status:
                    st.write("✅ Request sanitized (PII Gateway)")
                    st.write(f"⏳ Task queued (ID: {task_id[:8]}...)")
                    st.write("🤖 Medical Agent Team activated...")
                    
                    # Poll every 2 seconds
                    while True:
                        status_resp = requests.get(f"{API_URL}/tasks/{task_id}")
                        
                        if status_resp.status_code == 200:
                            job_info = status_resp.json()
                            job_status = job_info["status"]
                            
                            if job_status == "complete":
                                status.update(label="Response Ready!", state="complete", expanded=False)
                                final_answer = job_info["result"]
                                break
                            
                            elif job_status == "failed":
                                status.update(label="Task Failed", state="error")
                                st.error(f"Agents encountered an error: {job_info.get('error')}")
                                final_answer = None
                                break
                            
                            else:
                                # Still queued or in_progress
                                time.sleep(1.5)
                        else:
                            st.error("Failed to check task status.")
                            break
                
                # C. Display Final Result
                if final_answer:
                    message_placeholder.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
            else:
                st.error(f"Failed to enqueue task. API returned: {enqueue_resp.status_code}")

        except Exception as e:
            st.error(f"Frontend Logic Error: {e}")