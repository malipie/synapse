import streamlit as st
import requests
import os
import time
import json

# --- Configuration ---
API_URL = os.getenv("API_URL", "http://synapse-backend:8000/api/v1")

st.set_page_config(page_title="Synapse Enterprise", page_icon="🧠", layout="wide")

st.title("Synapse 🧠 ⟷ 🤖")
st.markdown("### Enterprise Medical RAG & Agentic Platform")

# --- Sidebar ---
with st.sidebar:
    st.header("🗂️ Document Ingestion")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file:
        if st.button("Ingest Document"):
            with st.spinner("Uploading & Processing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    response = requests.post(f"{API_URL}/documents/ingest", files=files)
                    if response.status_code == 200:
                        st.success("Document processed!")
                        st.json(response.json())
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
    st.divider()

# --- Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # 1. Wyślij zapytanie
            payload = {"messages": [{"role": "user", "content": prompt}], "model": "gpt-3.5-turbo"}
            response = requests.post(f"{API_URL}/chat/", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                intent = data.get("intent")
                
                # A. Szybka odpowiedź (Small Talk)
                if intent == "CHAT":
                    content = data.get("content")
                    message_placeholder.markdown(content)
                    st.session_state.messages.append({"role": "assistant", "content": content})
                
                # B. Długa odpowiedź (RAG) - Tu wchodzi Polling
                elif intent == "RAG":
                    job_id = data.get("job_id")
                    
                    with st.status("🕵️‍♂️ Agents are thinking...", expanded=True) as status:
                        st.write("✅ Request queued")
                        st.write(f"🆔 Job ID: {job_id}")
                        
                        # PĘTLA POLLINGU
                        while True:
                            time.sleep(2) # Czekaj 2 sekundy
                            # Pytamy o status zadania
                            status_resp = requests.get(f"{API_URL}/chat/tasks/{job_id}")
                            
                            if status_resp.status_code == 200:
                                job_data = status_resp.json()
                                job_status = job_data.get("status")
                                
                                if job_status == "complete":
                                    result = job_data.get("result")
                                    status.update(label="Done!", state="complete", expanded=False)
                                    # Wyświetl wynik końcowy
                                    message_placeholder.markdown(result)
                                    st.session_state.messages.append({"role": "assistant", "content": result})
                                    break
                                elif job_status == "in_progress" or job_status == "queued":
                                    continue # Czekaj dalej
                                else:
                                    status.update(label="Failed", state="error")
                                    st.error("Task failed.")
                                    break
                            else:
                                st.error(f"Status check failed: {status_resp.status_code}")
                                break
            else:
                st.error(f"API Error {response.status_code}")
                st.json(response.json())

        except Exception as e:
            st.error(f"Frontend Logic Error: {e}")