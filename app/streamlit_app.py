"""Streamlit demo UI. Run: streamlit run app/streamlit_app.py"""
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schemas import Candidate  # noqa: E402
from src.service import generate_application  # noqa: E402

st.set_page_config(page_title="CareerForge", page_icon="🛠️", layout="wide")
st.title("🛠️ CareerForge")
st.caption("Agentic, fact-grounded CV generator: retrieve evidence, write, critique, repeat.")

with st.sidebar:
    st.header("Candidate")
    name = st.text_input("Full name")
    email = st.text_input("Email")
    phone = st.text_input("Phone")
    location = st.text_input("Location")
    links = st.text_input("Links (comma-separated)")
    st.header("Options")
    use_critic = st.checkbox("Enable Critic loop", value=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Your material")
    uploads = st.file_uploader(
        "Old CVs, reports, notes (pdf, docx, md, txt)",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
    )
    github_user = st.text_input("GitHub username (optional)")
    extra_text = st.text_area("Or paste text about your experience", height=120)
with col2:
    st.subheader("2. Job offer")
    job_text = st.text_area("Paste the job offer", height=300)

if st.button("Generate tailored CV", type="primary", disabled=not (name and job_text)):
    with st.spinner("Agents at work (analyze → retrieve → write → critique)..."):
        tmp = Path(tempfile.mkdtemp(prefix="cf_ui_"))
        paths = []
        for f in uploads or []:
            p = tmp / Path(f.name).name
            p.write_bytes(f.getvalue())
            paths.append(p)
        try:
            res = generate_application(
                Candidate(
                    name=name,
                    email=email,
                    phone=phone,
                    location=location,
                    links=[x.strip() for x in links.split(",") if x.strip()],
                ),
                job_text,
                paths,
                github_user,
                extra_text,
                use_critic,
            )
        except Exception as e:  # show the error instead of a stack trace in the demo
            st.error(f"Generation failed: {e}")
            st.stop()

    s = res["scores"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Faithfulness", f"{s['faithfulness']:.0%}")
    m2.metric("ATS score", f"{s['ats']:.0%}")
    m3.metric("Quality", f"{s['quality']:.0%}")
    m4.metric("Overall", f"{s['overall']:.0%}")
    m5.metric("Iterations", res["iterations"])

    tab_cv, tab_loop, tab_letter, tab_q = st.tabs(
        ["CV", "Critic loop", "Cover letter", "Interview questions"]
    )
    with tab_cv:
        st.markdown(res["markdown"])
        out = Path(res["out_dir"])
        d1, d2, d3 = st.columns(3)
        d1.download_button("Download .tex", (out / "cv.tex").read_bytes(), "cv.tex")
        d2.download_button("Download .md", (out / "cv.md").read_bytes(), "cv.md")
        if res["pdf"]:
            d3.download_button("Download PDF", (out / "cv.pdf").read_bytes(), "cv.pdf")
        else:
            d3.info("PDF needs pdflatex/tectonic (use Docker).")
    with tab_loop:
        if res["history"]:
            st.dataframe(pd.DataFrame(res["history"]).drop(columns=["unsupported", "missing_keywords"]))
            st.line_chart(
                pd.DataFrame(res["history"]).set_index("iteration")[["faithfulness", "ats", "quality"]]
            )
        else:
            st.write("Critic loop was disabled for this run.")
    with tab_letter:
        st.write(res["cover_letter"])
    with tab_q:
        st.write(res["questions"])
