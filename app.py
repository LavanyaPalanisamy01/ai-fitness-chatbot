import os
import streamlit as st

from fitness import build_system_prompt, calculate_bmi

# Configuration for local Ollama instance
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3.2:3b"

# Configuration for cloud Hugging Face model
HF_MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"

# Detect Hugging Face token from Streamlit secrets or environment
hf_token = None
try:
    if "HF_TOKEN" in st.secrets:
        hf_token = st.secrets["HF_TOKEN"]
except Exception:
    pass

if not hf_token:
    hf_token = os.getenv("HF_TOKEN")

st.set_page_config(page_title="AI Fitness Chatbot", page_icon="💪", layout="centered")

# --- Sidebar: User Profile & BMI ---
st.sidebar.header("👤 User Profile")

age = st.sidebar.number_input("Age", min_value=10, max_value=120, value=25, step=1)
height = st.sidebar.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.5)
weight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.5)

goal = st.sidebar.selectbox(
    "Fitness Goal",
    options=["Weight Loss", "Muscle Gain", "General Fitness"]
)

# Calculate and display BMI
bmi, bmi_category = calculate_bmi(weight, height)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Body Mass Index (BMI)")
st.sidebar.metric(label="Your BMI", value=f"{bmi}")
st.sidebar.info(f"Classification: **{bmi_category}**")

# Reset chat button
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Reset Chat"):
    st.session_state.messages = []
    st.rerun()

# Display active engine
if hf_token:
    st.sidebar.caption(f"☁️ Cloud Model: `{HF_MODEL}` (Hugging Face)")
else:
    st.sidebar.caption(f"🤖 Local Model: `{OLLAMA_MODEL}` (Ollama)")

# --- Main Screen: Chatbot ---
st.title("💪 AI Fitness Chatbot")
st.caption(f"Profile: Age {age} • Goal: **{goal}** • BMI: **{bmi} ({bmi_category})**")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User chat input
if prompt := st.chat_input("Ask about workout routines, nutrition advice, or fitness tips..."):
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build system prompt with profile and BMI context
    system_prompt = build_system_prompt(
        age=int(age),
        height_cm=float(height),
        weight_kg=float(weight),
        goal=str(goal),
        bmi=float(bmi),
        bmi_category=str(bmi_category),
    )

    # Prepare message payload with system context and session history
    messages_payload = [{"role": "system", "content": system_prompt}] + st.session_state.messages

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            if hf_token:
                # Cloud deployment: Hugging Face InferenceClient
                from huggingface_hub import InferenceClient

                hf_client = InferenceClient(model=HF_MODEL, token=hf_token)
                response_stream = hf_client.chat.completions.create(
                    messages=messages_payload,
                    stream=True,
                    max_tokens=1024,
                )
                for chunk in response_stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        full_response += token
                        response_placeholder.markdown(full_response + "▌")
            else:
                # Local development: Ollama
                import ollama

                client = ollama.Client(host=OLLAMA_HOST)
                response_stream = client.chat(
                    model=OLLAMA_MODEL,
                    messages=messages_payload,
                    stream=True,
                )
                for chunk in response_stream:
                    token = chunk.get("message", {}).get("content", "")
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            response_placeholder.empty()
            provider_label = "Hugging Face Cloud" if hf_token else f"Local Ollama server at {OLLAMA_HOST}"
            st.error(f"Could not connect to {provider_label}. Details: {e}")
