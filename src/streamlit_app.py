import os
import tempfile
import pathlib
import json
import re
from datetime import datetime
os.environ["STREAMLIT_CONFIG_DIR"] = os.path.join(tempfile.gettempdir(), ".streamlit")
pathlib.Path(os.environ["STREAMLIT_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

import threading
import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

# ---------- CACHE PATHS (robust & writable) ----------
def pick_cache_root() -> str:
    # 1) Try the user's home (~/.cache/huggingface)
    home = pathlib.Path.home() / ".cache" / "huggingface"
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / ".write_test"
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
        return str(home)
    except Exception:
        pass
    # 2) Fall back to /tmp (always writable in HF Spaces)
    tmp = pathlib.Path(tempfile.gettempdir()) / "huggingface"
    tmp.mkdir(parents=True, exist_ok=True)
    return str(pathlib.Path(tempfile.gettempdir()) / "huggingface")

CACHE_ROOT = pick_cache_root()
HF_HUB_CACHE = os.path.join(CACHE_ROOT, "hub")
TRANSFORMERS_CACHE = os.path.join(CACHE_ROOT, "transformers")

os.environ["HF_HOME"] = CACHE_ROOT
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

pathlib.Path(HF_HUB_CACHE).mkdir(parents=True, exist_ok=True)
pathlib.Path(TRANSFORMERS_CACHE).mkdir(parents=True, exist_ok=True)

# ---------- UI CONFIG ----------
st.set_page_config(page_title="Tiny Llama Chef", page_icon="🤖", layout="centered")
st.title("🤖  LlamaChef")
st.subheader("Hello!, I am a Chef model. Give me ingredient, I will give you the recipe.")
if "submissions" not in st.session_state:
    st.session_state.submissions = [] 
with st.sidebar:
    st.markdown("### Known Output")
    # init storage
    if "known_outputs" not in st.session_state:
        st.session_state.known_outputs = []
    if st.session_state["submissions"]:
    # Shape the JSON exactly as you requested
        export = [
            {"email_address": s["email_address"], "rating": s["rating"]}
            for s in st.session_state["submissions"]
        ]
        json_bytes = json.dumps(export, indent=2).encode("utf-8")
        fname = f"ratings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        st.download_button(
            label="⬇️ Download ratings JSON",
            data=json_bytes,
            file_name=fname,
            mime="application/json",
        )
    # input + submit
    with st.form("known_output_form", clear_on_submit=True):
        ko = st.text_area(
            "Enter known output",
            height=120,
            help="Paste or type the expected output you already know.",
        )
        submitted = st.form_submit_button("Submit")

    if submitted and ko.strip():
        st.session_state.known_outputs.append(ko.strip())
        st.success("Saved!")

    # show saved outputs under the input box
    if st.session_state.known_outputs:
        st.markdown("#### Saved known outputs")
        # show most recent first
        for idx, text in enumerate(st.session_state.known_outputs, start=1):
            st.markdown(f"**#{idx}**")
            st.write(text)
            st.divider()

        # optional: small controls
        col1, col2 = st.columns(2)
        if col1.button("Clear last"):
            if st.session_state.known_outputs:
                st.session_state.known_outputs.pop()
                st.rerun()
        if col2.button("Clear all"):
            st.session_state.known_outputs = []
            st.rerun()
    else:
        st.caption("No known outputs yet.")

# ---------- MODEL LOADING (cached) ----------
@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
        cache_dir=TRANSFORMERS_CACHE,  # <-- important
    )
    is_cuda = torch.cuda.is_available()
    device_map = "auto" if is_cuda else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        trust_remote_code=True,
        cache_dir=TRANSFORMERS_CACHE,  # <-- important
    )
    return tok, model
model_id="Ragehoof/Tiny_bit4"
# Only (re)load when model_id changes
tokenizer, model = load_model_and_tokenizer(model_id)

# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # each: {role:"user"|"assistant", content:str}
if "ratings" not in st.session_state:
    st.session_state.ratings = {}
 
# ---------- HELPERS ----------
def render_history():
    for i, m in enumerate(st.session_state["messages"]):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

            if m["role"] == "assistant":
                # Has this assistant message already been submitted?
                already = any(s.get("message_index") == i for s in st.session_state["submissions"])
                if already:
                    # Show saved rating if available
                    r = next((s["rating"] for s in st.session_state["submissions"] if s["message_index"] == i), None)
                    if r is not None:
                        st.caption(f"⭐ Rating submitted: {r}/10")
                    # Show saved email too
                    e = next((s["email_address"] for s in st.session_state["submissions"] if s["message_index"] == i), None)
                    if e:
                        st.caption(f"📧 Email: {e}")
                else:
                    # Form to submit email + rating (won't save until user clicks submit)
                    with st.form(f"rate_email_form_{i}", clear_on_submit=False):
                        st.markdown("**Submit your feedback**")
                        email_val = st.text_input(
                            "Email address",
                            key=f"email_{i}",
                            placeholder="you@example.com",
                        )
                        rating_val = st.slider(
                            "Rating (1–10)",
                            min_value=1,
                            max_value=10,
                            value=8,
                            key=f"rate_{i}",
                        )
                        submitted = st.form_submit_button("Submit")

                    if submitted:
                        # very light validation
                        if "@" not in email_val or "." not in email_val:
                            st.error("Please enter a valid email address.")
                        else:
                            # store submission
                            st.session_state["submissions"].append({
                                "message_index": i,
                                "email_address": email_val.strip(),
                                "rating": int(rating_val),
                            })
                            # (optional) also mirror into ratings dict for quick lookup
                            st.session_state["ratings"][i] = int(rating_val)
                            st.success("Thanks! Your feedback has been saved.")
                            st.rerun()
SYSTEM = {
    "role": "system",
    "content": (
        "You are RecipeBot. Always answer using ONLY the latest ingredients the user provides. "
        "Ignore previous ingredient lists unless the user explicitly says to combine them. "
        "Keep the conversation context otherwise."
    ),
}
def build_chat_prompt(messages):
    try:
        return tokenizer.apply_chat_template(
            [SYSTEM] + messages,  # prepend system message
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # fallback simple concat
        turns = [f"### {m['role'].capitalize()}\n{m['content']}" for m in [SYSTEM] + messages]
        turns.append("### Assistant\n")
        return "\n\n".join(turns)

def stream_generate(prompt_text: str):
    """Token-stream the model output for Streamlit's write_stream."""
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    temperature =0.1
    top_p = 0.75
    max_new_tokens = 512
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=float(temperature),
        top_p=float(top_p),
        do_sample=(temperature > 0.0),
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    for text in streamer:
        yield text
    thread.join()

# ---------- RENDER HISTORY ----------
render_history()

# ---------- CHAT INPUT ----------
INGR_PAT = re.compile(r"\bingredient|\[.*\]|^\s*[-•]\s", re.IGNORECASE | re.DOTALL)

if prompt := st.chat_input("Say something…"):
    
    # 🔁 Keep context, but ensure only the *latest* ingredient list is present
    if INGR_PAT.search(prompt):
        st.session_state["messages"] = [
            m for m in st.session_state["messages"]
            # drop earlier ingredient-style user turns, keep everything else
            if not (m.get("role") == "user" and INGR_PAT.search(m.get("content", "")))
        ]

    # now add the new user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build prompt with system + full (filtered) history
    prompt_text = build_chat_prompt(st.session_state["messages"])

    with st.chat_message("assistant"):
        response_text = st.write_stream(stream_generate(prompt_text))

    st.session_state["messages"].append({"role": "assistant", "content": response_text})
    st.rerun()
if st.session_state.get("messages"):  # True if list not empty
    if st.button("🧹 Clear chat"):
        st.session_state["messages"] = []
        st.session_state["ratings"] = {}
        st.session_state["submissions"] = []
        st.rerun()
    
