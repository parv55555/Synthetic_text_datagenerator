import streamlit as st
import pandas as pd
import json
import io
import random
import os
import asyncio
import sqlite3

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from backend import get_api_key_env_var, async_generate_single_example, async_evaluate_with_llm

st.set_page_config(page_title="Synthetic Text Data Generator", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    /* Global Typography & Hide Defaults */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Background & Main App */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }

    /* Sidebar Aesthetic */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Force Text Colors to be Light */
    .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp label, .stApp span, .stApp div {
        color: #f8fafc;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
    }

    /* Gradient Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(168, 85, 247, 0.6);
        background: linear-gradient(135deg, #a855f7 0%, #d946ef 100%);
    }

    /* Glassmorphism Metrics */
    [data-testid="stMetricValue"] {
        font-weight: 800 !important;
        color: #f8fafc !important;
        font-size: 2.2rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600 !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.85rem !important;
    }
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        transition: transform 0.3s ease;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.3);
    }

    /* Titles */
    .main-title {
        text-align: center;
        background: linear-gradient(to right, #6366f1, #d946ef);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0px;
        letter-spacing: -1px;
    }
    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        font-weight: 300;
        margin-bottom: 40px;
        letter-spacing: 0.5px;
    }
    
    /* Inputs & Textareas */
    .stTextInput>div>div>input, .stTextArea>div>textarea {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>textarea:focus {
        border-color: #a855f7 !important;
        box-shadow: 0 0 0 1px #a855f7 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- UI Sidebar (API Keys & Config) ---
st.sidebar.title("API Keys & Configuration")
st.sidebar.markdown("This tool uses a **Bring Your Own Key** (BYOK) approach.")

with st.sidebar.expander("API Keys", expanded=True):
    api_keys = {
        "OpenAI": st.text_input("OpenAI API Key", type="password"),
        "Anthropic": st.text_input("Anthropic API Key", type="password"),
        "Google": st.text_input("Google Gemini API Key", type="password"),
        "Groq": st.text_input("Groq API Key", type="password"),
        "Mistral": st.text_input("Mistral API Key", type="password"),
        "Together": st.text_input("Together AI API Key", type="password"),
    }

st.sidebar.subheader("Model Configuration")
PROVIDERS = ["OpenAI", "Anthropic", "Google", "Groq", "Mistral", "Together"]

gen_provider = st.sidebar.selectbox("Generation Provider", PROVIDERS, index=3) # Default Groq
gen_model = st.sidebar.text_input("Generation Model (e.g. groq/llama3-8b-8192)", value="groq/llama3-8b-8192")

eval_provider = st.sidebar.selectbox("Evaluation Provider", PROVIDERS, index=2) # Default Google
eval_model = st.sidebar.text_input("Evaluation Model (e.g. gemini/gemini-1.5-flash)", value="gemini/gemini-1.5-flash")

task_description = st.sidebar.text_area("Task Description", value="Generate a realistic SMS or chat message.")
intent_list = st.sidebar.text_input("Categories/Labels (comma-separated)", value="scam, safe")
classes = [c.strip() for c in intent_list.split(",") if c.strip()]
if not classes:
    classes = ["scam", "safe"]
num_examples = st.sidebar.number_input("Number of Examples to Generate", min_value=1, max_value=1000, value=10)

st.sidebar.markdown("**Class Distribution Weights**")
class_weights_input = {}
for c in classes:
    class_weights_input[c] = st.sidebar.slider(f"Weight for '{c}'", 0, 100, int(100/len(classes)), key=f"weight_{c}")

total_weight = sum(class_weights_input.values())
if total_weight == 0:
    class_weights = {c: 1.0/len(classes) for c in classes}
else:
    class_weights = {c: w/total_weight for c, w in class_weights_input.items()}

target_counts = {c: int(num_examples * class_weights[c]) for c in classes}
while sum(target_counts.values()) < num_examples:
    target_counts[classes[0]] += 1
while sum(target_counts.values()) > num_examples:
    target_counts[classes[0]] -= 1
temperature = st.sidebar.slider("Temperature (Creativity)", min_value=0.0, max_value=2.0, value=0.85, step=0.05)
use_diversity = st.sidebar.checkbox("Enable Diversity/Persona Engine", value=True, help="Automatically injects diverse tones and personas into the prompt to prevent repetitive data.")

# --- Async Pipeline Runner ---
async def run_single_pipeline(intent, context, schema_def, few_shot):
    try:
        gen_json, gen_cost = await async_generate_single_example(
            gen_provider, gen_model, task_description, intent, context, schema_def, few_shot, temperature, use_diversity
        )
        
        if not gen_json:
            return {"success": False, "cost": gen_cost}
            
        eval_result = await async_evaluate_with_llm(
            eval_provider, eval_model, gen_json, intent, task_description
        )
        
        score = eval_result.get("score", 1)
        reasoning = eval_result.get("reasoning", "")
        eval_cost = eval_result.get("cost", 0.0)
        
        is_valid = score >= 4
        return {
            "success": is_valid,
            "data": gen_json,
            "score": score,
            "reasoning": reasoning,
            "cost": gen_cost + eval_cost,
            "intent": intent
        }
    except Exception as e:
        return {"success": False, "error": str(e), "cost": 0.0}

# --- Main App ---
st.markdown("<div class='main-title'>🧠 Structured Synthetic Data Generator</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Generate high-quality JSON datasets based on schemas provided in uploaded PDFs/TXTs using RAG context injection.</div>", unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📄 1. Provide Target Schema")
    st.info("You can define the schema directly or upload a context document.", icon="ℹ️")
    schema_def = st.text_area("JSON Schema Definition (e.g. {\"text\": \"string\"})", value='{\n  "text": "string"\n}')
    few_shot = st.text_area("Few-Shot Examples (Optional)", value="")
    uploaded_file = st.file_uploader("Upload Context Document (PDF/TXT) Optional", type=["pdf", "txt"])

with col2:
    st.markdown("### 🚀 2. Execute Pipeline")
    st.markdown("Ensure your API keys and configuration in the sidebar are correct before generating.")
    generate_btn = st.button("Generate Dataset", use_container_width=True)

if generate_btn:
    # Set environment variables for litellm
    for provider, key in api_keys.items():
        if key:
            os.environ[get_api_key_env_var(provider)] = key
            
    if not api_keys.get(gen_provider):
        st.error(f"Please provide the API key for the selected Generation Provider ({gen_provider}).")
        st.stop()
    if not api_keys.get(eval_provider):
        st.error(f"Please provide the API key for the selected Evaluation Provider ({eval_provider}).")
        st.stop()
    if not schema_def.strip() and not uploaded_file:
        st.warning("⚠️ Please provide a schema definition or upload a context document!")
        st.stop()
        
    st.toast("🚀 Pipeline started!", icon="🔥")
    
    with st.status("Running LLM-as-a-judge Pipeline...", expanded=True) as status_box:
        st.write("Initializing Context Engine...")
        full_context = "" 
        if uploaded_file:
            temp_path = f"temp_{uploaded_file.name}"
            try:
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(temp_path)
                else:
                    loader = TextLoader(temp_path, encoding='utf-8')
                    
                docs = loader.load()
                full_context = "\n".join([doc.page_content for doc in docs])
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
        st.write("Context loaded. Starting generation loop...")
        progress_bar = st.progress(0)
        
        max_attempts = num_examples * 5
        attempts = 0
        generated_data_local = []
        total_pipeline_cost = 0.0
        
        async def main_loop():
            global attempts, total_pipeline_cost
            
            while len(generated_data_local) < num_examples and attempts < max_attempts:
                current_counts = {c: len([d for d in generated_data_local if d["label"] == c]) for c in classes}
                needed_counts = {c: max(0, target_counts[c] - current_counts[c]) for c in classes}
                total_needed = sum(needed_counts.values())
                
                if total_needed == 0:
                    break
                    
                intents_to_spawn = []
                for c in classes:
                    if needed_counts[c] > 0:
                        spawn_count = needed_counts[c] + (needed_counts[c] // 2) + 1
                        intents_to_spawn.extend([c] * spawn_count)
                        
                random.shuffle(intents_to_spawn)
                intents_to_spawn = intents_to_spawn[:20] # cap batch size at 20
                batch_size = len(intents_to_spawn)
                
                st.write(f"Generating asynchronous batch of {batch_size} examples (Attempt {attempts + 1})...")
                
                tasks = []
                for intent in intents_to_spawn:
                    tasks.append(run_single_pipeline(intent, full_context, schema_def, few_shot))
                
                results = await asyncio.gather(*tasks)
                attempts += batch_size
                
                for r in results:
                    total_pipeline_cost += r.get("cost", 0.0)
                    if r.get("success"):
                        intent = r.get("intent")
                        current_count = len([d for d in generated_data_local if d["label"] == intent])
                        if current_count < target_counts[intent] and len(generated_data_local) < num_examples:
                            final_record = r["data"].copy()
                            final_record["label"] = intent
                            final_record["eval_score"] = r.get("score")
                            final_record["eval_reasoning"] = r.get("reasoning")
                            generated_data_local.append(final_record)
                            
                progress_bar.progress(min(len(generated_data_local) / num_examples, 1.0))
                
        # Run the async loop
        asyncio.run(main_loop())
            
        if len(generated_data_local) == num_examples:
            status_box.update(label="Generation Complete!", state="complete", expanded=False)
            st.toast("✅ Dataset generation successful!", icon="🎉")
        else:
            status_box.update(label=f"Stopped early. Generated {len(generated_data_local)}/{num_examples}.", state="error", expanded=False)
            st.toast("⚠️ Pipeline stopped early due to max attempts.", icon="⚠️")
    
    if generated_data_local:
        st.markdown("### 📊 Dataset Output")
        df = pd.DataFrame(generated_data_local)
        
        # Show some quick metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Generated", len(generated_data_local))
        if attempts > 0:
            success_rate = (len(generated_data_local) / max(1, attempts)) * 100
            rate_str = f"{success_rate:.1f}%"
        else:
            rate_str = "0%"
        m2.metric("Success Rate", rate_str)
        m3.metric("Total Pipeline Cost", f"${total_pipeline_cost:.4f}")
        
        avg_score = df["eval_score"].mean() if "eval_score" in df.columns else 0.0
        m4.metric("Avg Quality Score", f"{avg_score:.2f} / 5.0")
        
        st.dataframe(df, use_container_width=True)
        
        st.markdown("### 📈 Visualizations")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.markdown("**Category Distribution**")
            st.bar_chart(df['label'].value_counts())
        with v_col2:
            st.markdown("**Field Keys Density**")
            st.bar_chart(df.count())
        
        # Prepare Downloads
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        jsonl_data = "\n".join([json.dumps(row) for row in generated_data_local])
        
        st.markdown("### 💾 Export")
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button("Download as CSV", data=csv_data, file_name="synthetic_dataset.csv", mime="text/csv", use_container_width=True)
        with dl2:
            st.download_button("Download as JSONL", data=jsonl_data, file_name="synthetic_dataset.jsonl", mime="application/jsonl", use_container_width=True)
        with dl3:
            with st.popover("💾 Save to SQLite"):
                db_name = st.text_input("Database Name", value="synthetic_data.db")
                table_name = st.text_input("Table Name", value="dataset")
                if st.button("Save to DB"):
                    try:
                        conn = sqlite3.connect(db_name)
                        df.to_sql(table_name, conn, if_exists="append", index=False)
                        conn.close()
                        st.success(f"Successfully saved {len(df)} rows to {db_name} -> {table_name}")
                    except Exception as e:
                        st.error(f"Error saving to database: {e}")
