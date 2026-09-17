import os
import re
import asyncio
import streamlit as st
from google import genai

# --- Config ---
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not GEMINI_KEY:
    st.error("GEMINI_API_KEY not set. Add it in Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=GEMINI_KEY)

MODELS_TO_TRY = ["gemini-3.5-flash", "gemini-flash-latest", "gemini-3.7-flash", "gemini-3.6-flash"]
MAX_WORDS = 3000
PARALLEL_BATCH = 4

BANNED_WORDS = [
    "essential","crucial","beneficial","impactful","furthermore","moreover",
    "consequently","accordingly","additionally","significantly","notably",
    "importantly","ultimately","nevertheless","nonetheless","conversely","novel",
    "robust","comprehensive","substantial","pivotal","paramount","indispensable",
    "seminal","delve","testament","showcase","underscore","leverage","foster",
    "garner","realm","multifaceted","nuanced","holistic","interplay","harness",
    "unlock","myriad","seamless","meticulous","intricate","tapestry","landscape",
    "groundbreaking","cutting-edge","state-of-the-art","transformative",
    "revolutionary","remarkable","exceptional","optimal","flawless","effortless",
    "intuitive","sophisticated","advanced","visionary","thoughtfully",
    "deliberately","systematically","comprehensively","robustly","rigorously",
    "precisely","accurately","appropriately","effectively","efficiently",
    "meaningfully","fundamentally","inherently","intrinsically","critically",
    "centrally","chiefly","principally","predominantly","largely","broadly",
    "generally","typically","commonly","frequently","often","usually",
    "consistently","reliably","uniformly",
]

BANNED_PHRASES = [
    "it's worth noting that","it's important to recognize","a key consideration",
    "a critical factor","the broader implications","the underlying mechanisms",
    "the interplay between","the complex relationship","a deeper understanding",
    "meaningful insights","actionable insights","a robust framework",
    "a comprehensive approach","a systematic analysis","an integrated framework",
    "a nuanced understanding","the multifaceted nature","a holistic approach",
    "a paradigm shift","at its core","in essence","in practice","in principle",
    "on a fundamental level","when considered together","taken as a whole",
    "from a broader perspective",
]

ACADEMIC_EXCEPTIONS = {
    "comprehensive","systematically","novel","significant","significantly",
    "robust","substantial","fundamentally","rigorously","precisely","accurately",
}

HUMANIZER_PROMPT = """You rewrite academic papers so they read as if a human researcher wrote them, not an AI. Apply all rules. Output only the rewritten text.

ABSOLUTE RULES:
1. Length: output must be within 10% of input length. Never shorter.
2. Preserve every fact, number, citation, and detail. No merging. No summarizing.
3. Do not add "we" unless it was in the original. Do not add framing sentences.
4. Output only the rewritten text. No preamble. No commentary.

RHYTHM:
- Within each paragraph: at least one sentence under 8 words, at least one over 20 words.
- No two consecutive sentences with word counts within 2 of each other.
- Do not write one-sentence paragraphs. Do not chop into fragments.

SENTENCE OPENINGS:
- Never begin two consecutive sentences with the same word.
- Never start with: "We don't...", "It's not just...", "X isn't just...", "This suggests...", "Importantly...", "Notably...", "Together, these...", "Overall...", "Taken together...", "Here's the thing...", "So what does this mean?", "But here's where it gets interesting...", "Put another way...", "The catch is..."

VOICE:
- Active voice. Past tense for methods/results. Present tense for knowledge/conclusions.
- No passive nominalization. "We assessed," not "the assessment of."

BANNED WORDS (never use):
essential, crucial, beneficial, impactful, furthermore, moreover, consequently, accordingly, additionally, notably, importantly, ultimately, nevertheless, nonetheless, conversely, pivotal, paramount, indispensable, seminal, delve, testament, showcase, underscore, leverage, foster, garner, realm, multifaceted, nuanced, holistic, interplay, harness, unlock, myriad, seamless, meticulous, intricate, tapestry, landscape, groundbreaking, cutting-edge, state-of-the-art, transformative, revolutionary, remarkable, exceptional, optimal, flawless, effortless, intuitive, sophisticated, visionary, thoughtfully, deliberately, comprehensively, robustly, rigorously, appropriately, effectively, efficiently, meaningfully, inherently, intrinsically, critically, centrally, chiefly, principally, predominantly, largely, broadly, generally, typically, commonly, frequently, often, usually, consistently, reliably, uniformly.

BANNED SUBSTITUTIONS:
- No "alongside" for "and." No "utilize" for "use." No "construct" for "create."
- No "demonstrate" for "show." No "facilitate" for "help." No "prior to" for "before."
- No "in order to" for "to."

BANNED PHRASES:
it's worth noting that, it's important to recognize, a key consideration, a critical factor, the broader implications, the underlying mechanisms, the interplay between, the complex relationship, a deeper understanding, meaningful insights, actionable insights, a robust framework, a comprehensive approach, a systematic analysis, an integrated framework, a nuanced understanding, the multifaceted nature, a holistic approach, a paradigm shift, at its core, in essence, in practice, in principle, on a fundamental level, when considered together, taken as a whole, from a broader perspective.

BANNED WIND-UPS:
X isn't just Y, X doesn't just Y, It's not just about X, This is more than X, It goes beyond X, What's interesting is...

LAZY PATTERNS:
It is important to note that, It should be emphasized that, It is worth mentioning that, Of note, It is clear that, There is a need for, Let me explain, Let me break this down, Here's the key point, In other words (as filler), That said (as hedge), There are several factors (without naming them), This is a complex issue (as dodge).

REDUNDANT PHRASES:
past history -> history, end result -> result, future plans -> plans, completely eliminate -> eliminate, very unique -> unique, basic fundamentals -> fundamentals.

OVERLY ACADEMIC PHRASES:
It is evident that, It can be seen that, It is apparent that, It is obvious that, There is no doubt that, The fact that, It is well known that, As is well known, It is widely accepted that.

FILLER WORDS TO CUT:
basically, actually, really, quite, very, just, simply.

PARAGRAPH STRUCTURE:
- Paragraphs: 4-7 sentences. Vary length. No one-sentence paragraphs.

FORMATTING:
- No markdown dividers. Plain text. Blank lines between paragraphs.

PATTERNS TO BREAK:
- No triads (three adjectives/examples/clauses in a row).
- Break parallel structure. Break symmetry.
- Never use "not only X but also Y."
- "While X, Y" max once per paragraph.
- No em dashes. No "By [verb]ing..." starts.
- No nominalizations. No "the [noun] of [noun]."

TONE:
Knowledgeable but approachable researcher. Direct. Clear. Not textbook. Not blogger. Not robot.

Now rewrite the text I send. Output only the rewritten text."""


def find_violations(text: str):
    found = []
    lower = text.lower()
    for word in BANNED_WORDS:
        if word in ACADEMIC_EXCEPTIONS:
            continue
        if re.search(r"\b" + re.escape(word) + r"\b", lower):
            found.append(word)
    for phrase in BANNED_PHRASES:
        if phrase in lower:
            found.append(phrase)
    return found


async def call_gemini(prompt: str) -> str:
    last_error = None
    for model_name in MODELS_TO_TRY:
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            last_error = e
            continue
    raise last_error


async def humanize_paragraph(text: str, max_retries: int = 2) -> str:
    prompt = f"{HUMANIZER_PROMPT}\n\n---\n\nTEXT TO REWRITE:\n\n{text}"
    output = await call_gemini(prompt)
    for _ in range(max_retries):
        violations = find_violations(output)
        if not violations:
            return output
        fix_prompt = (
            f"Previous output used banned words: {', '.join(violations)}. "
            f"Rewrite below. Replace every banned word with a plain alternative. "
            f"Keep all facts, numbers, citations. Output only the text.\n\n{output}"
        )
        output = await call_gemini(fix_prompt)
    return output


async def humanize_all(paragraphs: list) -> list:
    results = [None] * len(paragraphs)
    for i in range(0, len(paragraphs), PARALLEL_BATCH):
        batch = paragraphs[i:i + PARALLEL_BATCH]
        batch_results = await asyncio.gather(
            *[humanize_paragraph(p) for p in batch],
            return_exceptions=True,
        )
        for j, r in enumerate(batch_results):
            if isinstance(r, Exception):
                results[i + j] = paragraphs[i + j]
            else:
                results[i + j] = r
        await asyncio.sleep(2)
    return results


# --- Streamlit UI ---
st.set_page_config(page_title="Humanizer", page_icon="\u270d\ufe0f", layout="wide")
st.title("Humanizer")
st.caption("Academic paper rewriter - 3000 word limit")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input (AI text)")
    input_text = st.text_area("Paste text here", height=400, label_visibility="collapsed")
    word_count = len(input_text.split())
    st.caption(f"{word_count} / {MAX_WORDS} words")

with col2:
    st.subheader("Output (Humanized)")
    output_placeholder = st.empty()

run_btn = st.button("Humanize", type="primary", use_container_width=True)

if run_btn:
    if not input_text.strip():
        st.warning("Please paste some text.")
    elif word_count > MAX_WORDS:
        st.error(f"Text exceeds {MAX_WORDS} word limit ({word_count} words).")
    else:
        with st.spinner(f"Humanizing {word_count} words... this takes 30-60s"):
            raw_paras = input_text.split("\n\n")
            paragraphs = [p.strip().replace("\n", " ") for p in raw_paras if len(p.strip()) > 50]
            if not paragraphs:
                st.error("No valid paragraphs found.")
            else:
                results = asyncio.run(humanize_all(paragraphs))
                final = "\n\n".join(results)
                output_placeholder.text_area("Output", value=final, height=400, label_visibility="collapsed")
                st.caption(f"{len(final.split())} words out - {len(paragraphs)} paragraphs")
                st.download_button(
                    label="Download .txt",
                    data=final,
                    file_name="humanized.txt",
                    mime="text/plain",
                )
