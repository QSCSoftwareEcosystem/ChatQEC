"""Off-topic classifier (Haiku).

Cheap pre-filter that refuses non-QEC queries before reaching Sonnet.
Saves cost and adds an abuse guard (spec §12.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chatqec.conversation.memory import ConversationMemory
from chatqec.trace import trace

_SYS = (
    "You are a strict topic classifier for a Quantum Error Correction (QEC) assistant. "
    "Answer 'on_topic' if the question concerns QEC or its foundations: quantum codes "
    "(stabilizer, CSS, surface, color, repetition, qLDPC, ...), the STABILIZER FORMALISM "
    "(the stabilizer group/generators of a state or code, Pauli operators, the Pauli "
    "group, logical operators), syndrome extraction, decoders, fault tolerance, magic "
    "states, thresholds, or error models. A question asking for the stabilizer(s) of a "
    "specific state such as |0>, |1>, |+>, |-> or a Bell state IS on_topic. "
    "Also answer 'on_topic' for IBM or IQM QUANTUM HARDWARE questions (IQM is the "
    "ORNL IQM machine), since this assistant has dedicated tools for them: a "
    "backend's calibration/noise data (T1, T2, readout error, two-qubit or CZ gate "
    "error/fidelity, coupling map), which backends are available or their queue or "
    "health status, mapping/placing a quantum circuit (including a Bell state, GHZ "
    "state, QFT, or QAOA circuit) onto real IBM or IQM hardware, running it there, or "
    "checking a hardware job's result. These count as on_topic even without "
    "stabilizer/QEC-code language. "
    "Also answer 'on_topic' for FAULT-TOLERANT COMPILATION and the vendor QEC tools, "
    "which this assistant runs: compiling a circuit to Clifford+T, T count, gridsynth or "
    "Solovay-Kitaev rotation synthesis, Pauli-based computation (PBC), adding spacetime "
    "Pauli checks with postselection, comparing decoders (PyMatching, Tesseract, BP+OSD, "
    "MWPF, CUDA-Q), Deltakit memory experiments, Guppy/Selene programs, and the "
    "parameters of BB5, barbell, or tile codes. "
    "Also answer 'on_topic' for building a visualization link for a circuit in QuEra's "
    "Gemini Studio (bloqade.quera.com/studio, formerly the GLCB circuit builder, for "
    "QuEra's Gemini neutral-atom computer, not Google's Gemini), including T gadgets, and "
    "for QuEra's Bloqade QEC Studio: its examples, writing or checking a .deq program "
    "(CODE/GADGET/COMPOSE blocks), or building a QEC Studio link. "
    "You may be shown recent conversation turns before the new question, for context "
    "only — classify the NEW QUESTION, not the conversation as a whole. A short "
    "follow-up with no QEC keywords of its own (e.g. 'how many rounds?', 'why is that?') "
    "IS on_topic if the conversation it continues is about QEC. "
    "Answer 'off_topic' only if it is clearly unrelated to QEC, the stabilizer "
    "formalism, and IBM/IQM hardware (e.g. everyday trivia, general programming, or "
    "generic quantum-computing questions with no QEC/stabilizer/hardware content) "
    "and does not continue a QEC conversation. "
    "Respond with EXACTLY one word and nothing else: 'on_topic' or 'off_topic'. "
    "Do NOT answer the question or add any text after the label."
)

_DEFAULT_REFUSAL = (
    "ChatQEC only answers Quantum Error Correction questions. "
    "Please rephrase your question to focus on QEC, fault tolerance, "
    "stabilizer codes, decoders, or related topics."
)


@dataclass(frozen=True)
class TopicVerdict:
    on_topic: bool
    refusal_message: str = ""


class OffTopicClassifier:
    """Wraps an Anthropic client to perform single-token on/off-topic classification.

    The ``client`` parameter is typed as ``Any`` to avoid pinning the test surface to
    the full Anthropic SDK type; in production it is an ``anthropic.Anthropic`` instance.
    """

    def __init__(self, *, client: Any, model: str, history_turns: int = 6) -> None:
        self.client = client
        self.model = model
        self.history_turns = history_turns

    def classify(self, query: str, *, memory: ConversationMemory | None = None) -> TopicVerdict:
        recent = memory.recent(n=self.history_turns) if memory is not None else []
        if recent:
            history_block = "\n".join(f"{t.role}: {t.content}" for t in recent)
            content = f"Recent conversation:\n{history_block}\n\nNew question: {query}"
        else:
            content = query
        trace("classifier.call", model=self.model, query=query, history_turns=len(recent))
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=10,
            system=_SYS,
            messages=[{"role": "user", "content": content}],
        )
        text = str(msg.content[0].text).strip().lower()
        trace("classifier.response", model=self.model, raw=text)
        # Key off the leading label token, not exact-match: Haiku sometimes emits
        # the label and then keeps answering (e.g. "on_topic\n\nThe Steane code is …"
        # when the query says "give me an example"). Strip surrounding punctuation so
        # "on_topic." / "on_topic:" also count. A truly malformed first token
        # (anything but "on_topic") stays off_topic, which is the safe default.
        tokens = text.split()
        first = tokens[0].strip(".,:;!?\"'`") if tokens else ""
        if first == "on_topic":
            return TopicVerdict(on_topic=True)
        return TopicVerdict(on_topic=False, refusal_message=_DEFAULT_REFUSAL)
