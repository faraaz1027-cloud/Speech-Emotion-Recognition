# Original CSE424 analysis notebook

The source notebook and final coursework report are retained in the
[original team repository](https://github.com/waysal1/Speech-Emotion-Recognition).

- [Original notebook](https://github.com/waysal1/Speech-Emotion-Recognition/blob/main/CSE424_SER_Project_Code.ipynb)
- [Original PDF report](https://github.com/waysal1/Speech-Emotion-Recognition/blob/main/CSE424%20Project%20Report.pdf)

The user also supplied these files as a ZIP, but **the public portfolio
repository does not contain the primary dataset, external validation data,
or a fitted serialized classifier**. The notebook uses Google Drive-specific
data paths and saves an intermediate CSV in its Colab environment. Those
files must not be guessed or fabricated.

For privacy and repository size, a cleaned outputs-cleared notebook
can be uploaded separately after reviewing the original team's permissions
and dataset references. The Streamlit demo implements the notebook's
58 acoustic features from audios selected by visitors, without predicting
emotion.

**Methodological note:** The historical notebook fits its scaler to all
records before splitting or cross-validation; it must be reworked for
leakage-aware evaluation. Do not use its reported internal accuracy as
a claim of reproducible generalization.
