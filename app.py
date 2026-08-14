import streamlit as st

from core.excel_handler import (
    save_uploaded_file,
    load_excel,
    build_context,
    get_dataframe_summary,
)
from core.agent_engine import create_agent
from core.memory import create_memory, reset_memory

# Səhifə konfiqurasiyası
st.set_page_config(
    page_title="Sales AI Agent",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: white;
    }
    .sub-title {
        font-size: 0.9rem;
        color: #6b7280;
        margin-bottom: 1rem;
    }
    .file-info {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 0.75rem 1rem;
        border-radius: 0.375rem;
        font-size: 0.85rem;
        color: #166534;
    }
</style>
""", unsafe_allow_html=True)

# Session state 
if "memory"       not in st.session_state:
    st.session_state.memory = create_memory()
if "agent"        not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sheets"       not in st.session_state:
    st.session_state.sheets = {}
if "file_name"    not in st.session_state:
    st.session_state.file_name = None

# Sidebar
with st.sidebar:
    st.markdown("## 📁 Excel Faylı")

    uploaded = st.file_uploader(
        "Fayl seçin",
        type=["xlsx", "xls"],
        label_visibility="collapsed",
    )

    if uploaded and uploaded.name != st.session_state.file_name:
        with st.spinner("Fayl emal edilir..."):
            try:
                path   = save_uploaded_file(uploaded)
                sheets = load_excel(path)
                ctx    = build_context(sheets)

                st.session_state.memory       = create_memory()
                st.session_state.chat_history = []
                st.session_state.sheets       = sheets
                st.session_state.file_name    = uploaded.name

                st.session_state.agent = create_agent(
                    sheets=sheets,
                    memory=st.session_state.memory,
                    excel_context=ctx,
                )
                st.success("✅ Fayl uğurla yükləndi!")

            except Exception as e:
                st.error(f"❌ Xəta: {e}")

    if st.session_state.sheets:
        st.markdown("---")
        st.markdown("**📋 Overview**")
        st.markdown(
            f'<div class="file-info">'
            f'{get_dataframe_summary(st.session_state.sheets)}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    if st.button("🗑️ Söhbəti Sil", use_container_width=True):
        st.session_state.memory       = reset_memory()
        st.session_state.chat_history = []
        if st.session_state.sheets:
            ctx = build_context(st.session_state.sheets)
            st.session_state.agent = create_agent(
                sheets=st.session_state.sheets,
                memory=st.session_state.memory,
                excel_context=ctx,
            )
        st.rerun()

# Əsas sahə
st.markdown('<p class="main-title">📊 Sales AI Agent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Şirkətinizin kredit və satış məlumatlarını təhlil edən AI asistent</p>',
    unsafe_allow_html=True,
)

# Söhbət tarixçəsi
for role, content in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)

# Giriş
user_input = st.chat_input(
    "Sualınızı yazın...",
    disabled=(st.session_state.agent is None),
)

if user_input:
    if st.session_state.agent is None:
        st.warning("⚠️ Əvvəlcə sol paneldən Excel faylı yükləyin.")
    else:
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Düşünürəm..."):
                try:
                    response = st.session_state.agent.invoke({"input": user_input})
                    answer   = response.get("output", "Cavab alına bilmədi.")
                except Exception as e:
                    answer = f"❌ Xəta: {e}"

                st.markdown(answer)
                st.session_state.chat_history.append(("assistant", answer))

if st.session_state.agent is None:
    st.info("👈 Başlamaq üçün sol paneldən Excel faylı yükləyin.")