import matplotlib.pyplot as plt
import streamlit as st
from pyannote.audio.pipelines import (
    VoiceActivityDetection,
    OverlappedSpeechDetection,
)
from pyannote.audio import Audio
from pyannote.core import notebook, Segment
import io
import base64

from matplotlib.backends.backend_agg import RendererAgg

_lock = RendererAgg.lock

st.sidebar.image("https://avatars.githubusercontent.com/u/7559051?s=400&v=4")

st.markdown(
    """
# End-to-end speaker segmentation

This webapp demonstrates the _pyannote.audio_ [model](https://huggingface.co/pyannote/segmentation) introduced in  

> [End-to-end speaker segmentation for overlap-aware resegmentation](http://arxiv.org/abs/2104.04045)  
by Hervé Bredin and Antoine Laurent (submitted to Interspeech 2021)

Upload an audio file and its first 60 seconds will be processed automatically.
"""
)


TASKS = [
    # {"human-readable": "Speaker segmentation",
    #  "pipeline": Segmentation,
    #  "raw_scores": "@segmentation/activations",
    #  "activation": lambda data: data,
    #  "mapping": lambda labels: {label: f"speaker_{i+1:02d}" for i, label in enumerate(labels)},
    # },
    {
        "human-readable": "Voice activity detection",
        "pipeline": VoiceActivityDetection,
        "activation": "@voice_activity_detection/activation",
        "mapping": lambda labels: {label: "speech" for label in labels},
    },
    {
        "human-readable": "Overlapped speech detection",
        "pipeline": OverlappedSpeechDetection,
        "activation": "@overlapped_speech_detection/activation",
        "mapping": lambda labels: {label: "overlap" for label in labels},
    },
]

audio = Audio(sample_rate=16000, mono=True)


class ProgressHook:
    @property
    def progress_bar(self):
        return self._progress_bar

    @progress_bar.setter
    def progress_bar(self, progress_bar):
        self._progress_bar = progress_bar

    def __call__(self, chunk_idx, num_chunks):
        if chunk_idx >= num_chunks:
            self._progress_bar.empty()
        else:
            self._progress_bar.progress(chunk_idx / num_chunks)


progress_hook = ProgressHook()

st.sidebar.markdown(
    """
Use the model for...
"""
)

task = st.sidebar.selectbox(
    "", TASKS, index=0, format_func=lambda t: t["human-readable"], key="task"
)
Pipeline = task["pipeline"]

pipeline = Pipeline(
    segmentation="pyannote/segmentation", batch_size=1, progress_hook=progress_hook
)

more_options = st.sidebar.checkbox(
    "Give more options...", value=False, key="more_options"
)

if more_options:
    on, off = st.sidebar.beta_columns(2)
    onset = on.slider(
        "Onset threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.01,
        key="onset",
    )
    offset = off.slider(
        "Offset threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.01,
        key="offset",
    )
else:
    onset = 0.7
    offset = 0.3

hyper_parameters = {
    "onset": onset,
    "offset": offset,
    "min_duration_on": 0.0,
    "min_duration_off": 0.0,
}

pipeline.instantiate(hyper_parameters)

uploaded_file = st.file_uploader("")
if uploaded_file is not None:

    duration = audio.get_duration(uploaded_file)
    waveform, sample_rate = audio.crop(uploaded_file, Segment(0, min(duration, 60)))
    file = {"waveform": waveform, "sample_rate": sample_rate, "uri": uploaded_file.name}

    progress_bar = st.empty()
    progress_hook.progress_bar = progress_bar
    output = pipeline(file)
    output.rename_labels(mapping=task["mapping"](output.labels()), copy=False)

    with _lock:

        notebook.reset()
        notebook.crop = Segment(0, min(duration, 60))

        if more_options:
            scores = file[task["activation"]]

            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
            fig.set_figwidth(12)
            fig.set_figheight(5.0)
            notebook.plot_feature(scores, ax=ax1, time=False)
            ax1.plot(
                [0, duration - 1.4],
                [hyper_parameters["onset"], hyper_parameters["onset"]],
                "k--",
            )
            ax1.text(0.1, hyper_parameters["onset"] + 0.04, "onset")
            ax1.plot(
                [1.4, len(scores)],
                [hyper_parameters["offset"], hyper_parameters["offset"]],
                "k--",
            )
            ax1.text(
                min(duration, 60) - 1.1, hyper_parameters["offset"] + 0.04, "offset"
            )
            ax1.set_ylim(-0.1, 1.1)
            notebook.plot_annotation(output, ax=ax2, time=True, legend=True)

        else:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            fig.set_figwidth(12)
            fig.set_figheight(2.0)
            notebook.plot_annotation(output, ax=ax, time=True, legend=True)

            plt.tight_layout()
            st.pyplot(fig=fig, clear_figure=True)

    with io.StringIO() as fp:
        output.write_rttm(fp)
        content = fp.getvalue()

        b64 = base64.b64encode(content.encode()).decode()
        href = f'<a download="{output.uri}.rttm" href="data:file/text;base64,{b64}">Download as RTTM</a>'
        st.markdown(href, unsafe_allow_html=True)


st.sidebar.markdown(
    """
-------------------

To use this model on more and longer files on your own (GPU, hence much faster) servers, check the [documentation](https://huggingface.co/pyannote/segmentation).  

For [technical questions](https://github.com/pyannote/pyannote-audio/discussions) and [bug reports](https://github.com/pyannote/pyannote-audio/issues), please check [pyannote.audio](https://github.com/pyannote/pyannote-audio) Github repository.

For commercial enquiries and scientific consulting, please contact [me](mailto:herve@niderb.fr).
"""
)
