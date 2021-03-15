import base64
import io

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.backends.backend_agg import RendererAgg

from pyannote.audio.core.inference import Inference
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import notebook

_lock = RendererAgg.lock

st.markdown(
    """
# Voice activity detection
"""
)

vad = Inference(
    "hbredin/VoiceActivityDetection-PyanNet-DIHARD",
    batch_size=8,
    device="cpu",
)


pipeline = VoiceActivityDetection(scores="vad").instantiate(
    dict(
        min_duration_off=0.0010373822944547487,
        min_duration_on=0.028629859285383544,
        offset=0.3253816781523919,
        onset=0.8541166717794256,
    )
)

uploaded_file = st.file_uploader("Choose a file", type=['wav',])
if uploaded_file is not None:

    progress_bar = st.empty()

    def progress_hook(chunk_idx, num_chunks):
        progress_bar.progress(chunk_idx / num_chunks)

    vad.progress_hook = progress_hook
    with open(uploaded_file.name, "bw") as localfile:
        localfile.write(uploaded_file.read())
    scores = vad({"audio": uploaded_file.name})

    progress_bar.empty()

    with _lock:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_figwidth(12)
        fig.set_figheight(2.0)
        notebook.plot_feature(scores, ax=ax, time=True)
        ax.set_ylim(-0.1, 1.1)
        plt.tight_layout()
        st.pyplot(fig=fig, clear_figure=True)

    speech_regions = pipeline({"vad": scores, "uri": uploaded_file.name})

    with _lock:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_figwidth(12)
        fig.set_figheight(2.0)
        notebook.plot_timeline(speech_regions.get_timeline(), ax=ax, time=True)
        plt.tight_layout()
        st.pyplot(fig=fig, clear_figure=True)

    with io.StringIO() as fp:
        speech_regions.write_rttm(fp)
        content = fp.getvalue()

    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a download="vad.rttm" href="data:file/text;base64,{b64}">Download as RTTM</a>'
    st.markdown(href, unsafe_allow_html=True)
