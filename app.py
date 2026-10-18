# This was our basic frontend which we used to test our code

"""Streamlit front end for the Private Legal Triage Agent."""

import hashlib
import logging
import re

import ollama
import pymupdf
import streamlit as st

from legal_analyzer import (
    SINGLE_PASS_TOKENS,
    ReportError,
    create_conversation,
    estimate_tokens,
    follow_up,
)

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}
SEVERITY_ICON = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Private Legal Triage Agent",
    page_icon="⚖️",
    layout="wide",
)


# --------------------------------------------------
# PDF TEXT EXTRACTION
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def extract_pages(pdf_bytes: bytes):
    """Extract text per page. Returns (pages, skipped_page_numbers).

    Cached on the file's bytes, so Streamlit reruns don't re-extract.
    """
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as error:  # pymupdf raises several types for bad files
        raise ValueError("This file could not be opened as a PDF.") from error

    with document:
        if document.needs_pass:
            raise ValueError(
                "This PDF is password-protected. Remove the password and try again."
            )

        pages, skipped = [], []
        for number, page in enumerate(document, start=1):
            text = page.get_text(sort=True)  # sort=True: natural reading order
            if text.strip():
                pages.append({"page": number, "text": text})
            else:
                skipped.append(number)

    return pages, skipped


def build_document_text(pages: list[dict]) -> str:
    """Join pages with [Page N] markers, which legal_analyzer relies on."""
    return "\n\n".join(f"[Page {p['page']}]\n{p['text']}" for p in pages)


# --------------------------------------------------
# SOURCE HIGHLIGHTING
# --------------------------------------------------

def _token(word: str) -> str:
    """Lowercase a word and drop punctuation, so "Rs." and "rs" match."""
    return re.sub(r"\W+", "", word).casefold()


def find_quote_rects(page, quote: str) -> list:
    """Return one rectangle per text line covered by the quote ([] if not found)."""
    target = [t for t in map(_token, quote.split()) if t]
    if not target:
        return []

    # Each word: (token, rectangle, (block number, line number)).
    # sort=True matches the reading order used when the text was extracted.
    words = [
        (_token(w[4]), pymupdf.Rect(w[:4]), (w[5], w[6]))
        for w in page.get_text("words", sort=True)
    ]
    words = [w for w in words if w[0]]
    tokens = [w[0] for w in words]

    n = len(target)
    for i in range(len(tokens) - n + 1):
        if tokens[i : i + n] == target:
            lines = {}  # merge word boxes into one box per line (no gaps)
            for _, rect, key in words[i : i + n]:
                if key in lines:
                    lines[key].include_rect(rect)
                else:
                    lines[key] = pymupdf.Rect(rect)
            return list(lines.values())

    return page.search_for(quote)  # fallback: pymupdf's own text search


@st.cache_data(show_spinner=False)
def render_source(
    _pdf_bytes: bytes, doc_id: str, page_number, quote: str, full_page: bool = False
):
    """Return (page_number, PNG bytes) with the quote highlighted, or None.

    `_pdf_bytes` starts with an underscore so Streamlit doesn't hash the whole
    file on every call; `doc_id` identifies the document in the cache instead.
    """
    with pymupdf.open(stream=_pdf_bytes, filetype="pdf") as doc:
        # Page unknown (quote appears on several pages)? Try them all.
        candidates = [page_number - 1] if page_number else range(len(doc))

        for index in candidates:
            page = doc[index]
            rects = find_quote_rects(page, quote)
            if not rects:
                continue

            page.add_highlight_annot(rects)

            clip = page.rect
            if not full_page:  # crop to the passage plus some context
                area = pymupdf.Rect(rects[0])
                for r in rects[1:]:
                    area.include_rect(r)
                clip = pymupdf.Rect(
                    page.rect.x0,
                    max(page.rect.y0, area.y0 - 70),
                    page.rect.x1,
                    min(page.rect.y1, area.y1 + 70),
                )

            return index + 1, page.get_pixmap(dpi=130, clip=clip).tobytes("png")

    return None


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def show_error(error: Exception) -> None:
    """Turn exceptions into messages a non-technical user can act on."""
    if isinstance(error, (ReportError, ValueError)):
        message = str(error)
    elif isinstance(error, ConnectionError):
        message = (
            "Couldn't reach Ollama. Make sure it is running "
            "(`ollama serve`) and try again."
        )
    elif isinstance(error, ollama.ResponseError):
        if getattr(error, "status_code", None) == 404:
            message = "The model isn't installed. Pull it with `ollama pull <model>`."
        else:
            message = f"Ollama reported an error: {getattr(error, 'error', error)}"
    else:
        logger.error("Unexpected error", exc_info=error)
        message = f"Something went wrong: {error}"
    st.error(message)


def md_quote(text: str) -> str:
    """Render text as a safe markdown blockquote (escapes markdown/LaTeX chars)."""
    flat = re.sub(r"\s+", " ", text).strip()
    escaped = re.sub(r"([\\`*_{}\[\]()#+\-!|>~$])", r"\\\1", flat)
    return f"> {escaped}"


def render_report(report: dict, pdf_bytes=None, doc_id: str = "") -> None:
    """Display the structured report returned by legal_analyzer."""
    st.markdown("#### Summary")
    st.write(report["summary"])

    findings = sorted(
        report["findings"],
        key=lambda f: SEVERITY_ORDER.get(f["severity"], 3),
    )

    if not findings:
        st.info("No relevant risks were identified for this request.")
        return

    st.markdown(f"#### Findings ({len(findings)})")
    st.caption("Severity is a preliminary assessment, not a legal determination.")

    seen: dict[str, int] = {}  # keeps widget keys unique if two quotes are identical

    for finding in findings:
        severity = finding["severity"]
        label = f"{SEVERITY_ICON.get(severity, '⚪')} {severity} · {finding['title']}"

        with st.expander(label, expanded=(severity == "High")):
            meta = f"Category: {finding['category']}"
            if finding.get("page") is not None:
                meta += f" · Page {finding['page']}"
            st.caption(meta)

            st.markdown("**From the document**")
            st.markdown(md_quote(finding["quotation"]))

            if not finding.get("quote_verified", False):
                st.warning(
                    "Couldn't find this exact wording in the document. It may "
                    "be reworded or wrong. Check the original before relying on it."
                )
            elif pdf_bytes is not None:
                # Key is based on the quote (not the position), so it stays
                # stable when a follow-up changes the order of findings.
                digest = hashlib.sha256(finding["quotation"].encode()).hexdigest()[:12]
                count = seen.get(digest, 0)
                seen[digest] = count + 1
                key = f"src_{doc_id}_{digest}_{count}"

                if st.checkbox("Show in original document", key=key):
                    result = render_source(
                        pdf_bytes, doc_id, finding.get("page"), finding["quotation"]
                    )
                    if result:
                        page_no, png = result
                        st.image(png, caption=f"Page {page_no}, highlighted passage")
                    else:
                        st.caption(
                            "Couldn't locate this passage in the PDF layout "
                            "(it may span two pages)."
                        )

            st.markdown("**What it means**")
            st.write(finding["explanation"])

            st.markdown("**Suggested action**")
            st.write(finding["suggested_action"])


# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------

st.session_state.setdefault("messages", None)
st.session_state.setdefault("report", None)
st.session_state.setdefault("document_name", None)
st.session_state.setdefault("document_text", None)  # needed to verify quotes in follow-ups
st.session_state.setdefault("pdf_bytes", None)  # needed to highlight sources
st.session_state.setdefault("doc_id", "")
st.session_state.setdefault("follow_up_round", 0)  # changes the input's key to clear it


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("⚖️ Private Legal Triage Agent")

st.write(
    "Upload a legal document, analyse potential risks, "
    "and refine the results through follow-up prompts."
)

st.info(
    "Your document is analysed using your local Ollama model. "
    "Findings are preliminary and should not be treated as "
    "professional legal advice."
)


# --------------------------------------------------
# DOCUMENT UPLOAD AND INITIAL PROMPT
# --------------------------------------------------

uploaded_file = st.file_uploader("Upload a legal document", type=["pdf"])

initial_prompt = st.text_area(
    "Initial analysis prompt",
    value="Perform a general legal review of this document.",
    placeholder=(
        "Example: Focus on financial risks, hidden fees, "
        "and termination conditions."
    ),
    height=100,
)


# --------------------------------------------------
# ANALYSE DOCUMENT BUTTON
# --------------------------------------------------

if st.button("Analyse Document", type="primary"):

    if uploaded_file is None:
        st.error("Please upload a PDF document first.")

    elif not initial_prompt.strip():
        st.error("Please enter an initial analysis prompt.")

    else:
        try:
            pdf_bytes = uploaded_file.getvalue()

            with st.spinner("Extracting text from the PDF..."):
                pages, skipped = extract_pages(pdf_bytes)

            if not pages:
                st.error(
                    "No selectable text was found in this PDF. "
                    "It may be a scanned document that requires OCR."
                )

            else:
                if skipped:
                    st.warning(
                        f"No text found on page(s) {', '.join(map(str, skipped))}. "
                        "They may be scanned images and were left out of the analysis."
                    )

                document_text = build_document_text(pages)

                if estimate_tokens(document_text) > SINGLE_PASS_TOKENS:
                    st.info(
                        "This is a long document, so it will be analysed in "
                        "sections. That can take several minutes."
                    )

                st.success(f"Extracted text from {len(pages)} pages.")

                progress_bar = st.progress(
                    0.0, text="Gemma is analysing your document..."
                )

                def update_progress(done: int, total: int) -> None:
                    label = (
                        f"Analysing part {done + 1} of {total}..."
                        if done < total
                        else "Writing the summary..."
                    )
                    progress_bar.progress(done / total, text=label)

                try:
                    messages, report = create_conversation(
                        document_text,
                        initial_prompt.strip(),
                        progress=update_progress,
                    )
                finally:
                    progress_bar.empty()

                # Save everything across Streamlit reruns.
                st.session_state.messages = messages
                st.session_state.report = report
                st.session_state.document_name = uploaded_file.name
                st.session_state.document_text = document_text
                st.session_state.pdf_bytes = pdf_bytes
                st.session_state.doc_id = hashlib.sha256(pdf_bytes).hexdigest()[:16]
                st.session_state.follow_up_round += 1  # fresh follow-up box

                st.success("Analysis completed successfully!")

        except Exception as error:
            show_error(error)


# --------------------------------------------------
# DISPLAY INITIAL OR UPDATED REPORT
# --------------------------------------------------

if st.session_state.report is not None:

    st.divider()
    st.subheader("📄 Legal Triage Report")
    st.caption(f"Document: {st.session_state.document_name}")

    render_report(
        st.session_state.report,
        st.session_state.pdf_bytes,
        st.session_state.doc_id,
    )


# --------------------------------------------------
# FOLLOW-UP CONVERSATION
# --------------------------------------------------

if st.session_state.messages is not None:

    st.divider()
    st.subheader("💬 Refine Your Analysis")

    st.write(
        "Ask Gemma to revise the report, simplify explanations, "
        "focus on particular clauses, or add more detail."
    )

    if estimate_tokens(st.session_state.document_text or "") > SINGLE_PASS_TOKENS:
        st.caption(
            "Because this document was analysed in sections, follow-ups refine "
            "the existing findings. They don't re-read the full document."
        )

    follow_up_prompt = st.text_area(
        "Your follow-up prompt",
        placeholder=(
            "Example: Rewrite the report in simpler English "
            "and explain the financial risks in more detail."
        ),
        height=100,
        # A new key gives a fresh, empty box after a successful submit, while
        # a failed submit keeps the user's text. (Assigning to a widget's
        # key after it has been drawn raises an error in Streamlit.)
        key=f"follow_up_input_{st.session_state.follow_up_round}",
    )

    if st.button("Submit Follow-up", type="primary"):

        if not follow_up_prompt.strip():
            st.warning("Please enter a follow-up prompt.")

        else:
            try:
                with st.spinner("Gemma is updating your analysis..."):
                    # follow_up() already wraps the prompt in its own
                    # instructions, so pass the user's text as-is.
                    messages, updated_report = follow_up(
                        st.session_state.messages,
                        follow_up_prompt.strip(),
                        document_text=st.session_state.document_text,
                    )
            except Exception as error:
                show_error(error)
            else:
                st.session_state.messages = messages
                st.session_state.report = updated_report
                st.session_state.follow_up_round += 1
                st.rerun()


# --------------------------------------------------
# RESET CONVERSATION
# --------------------------------------------------

if st.session_state.messages is not None:

    if st.button("Start New Analysis"):
        st.session_state.messages = None
        st.session_state.report = None
        st.session_state.document_name = None
        st.session_state.document_text = None
        st.session_state.pdf_bytes = None
        st.session_state.doc_id = ""
        st.rerun()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Legal Triage Agent | Preliminary document review only. "
    "Verify findings against the original document and consult "
    "a qualified legal professional when appropriate."
)
