# Bangla Speech Emotion Recognition — CSE424

A **collaborative BRAC University coursework project** studying emotion-related
patterns in Bangla speech using acoustic features, clustering and supervised
machine learning. The portfolio repo includes a new **interactive acoustic
feature explorer** and a separate, untrained pipeline for later reproduction.

## Interactive portfolio demo

**Demo deployment:** Not published yet — no live URL is claimed here.

Try the Streamlit audio feature explorer after deploying `demo/app.py`.
It accepts short WAV recordings, shows a waveform and MFCC heatmap, and
calculates the same **58 acoustic descriptors** used in the coursework:
40 MFCC mean/std values, 12 chroma means, spectral centroid, bandwidth,
rolloff, zero-crossing rate, RMS energy and tempo.

**Crucial distinction:** The provided team ZIP does not include a dataset or
serialized trained classifier. **This app does not predict an emotion.**
Publishing a live emotion predictor requires properly sourced training data,
a re-fitted pipeline, leakage-aware evaluation and ideally speaker-independent
external testing. No audio uploaded to the demonstration is committed to
this repo or application database.

## Recorded coursework results (not independently reproduced)

| Classifier | Internal test accuracy | External balanced accuracy |
| --- | ---: | ---: |
| SVM | 76.54% | 33.6% |

The original notebook reports **five emotion classes**: Angry, Happy,
Natural (neutral), Sad, and Surprised. Its external validation is a balanced
set of **500 clips** (100 per class). Other internal models include Random
Forest, KNN, Logistic Regression and Gradient Boosting. The notebook also
explores K-Means and agglomerative clustering.

### Important methodological limitations

- The notebook fits `StandardScaler` on all examples **before splitting**
  into training and test groups, and cross-validation also uses these
  globally scaled inputs. Internal results may therefore be optimistic.
- The random stratified splits do not establish speaker independence.
- Performance drops to 33.6% external SVM accuracy. This is an important
  limitation; the research should not be described as production-validated.
- The supplied ZIP contains no raw training clips and no fitted `.joblib`
  or other model artifact. Historical results have not been rerun here.
- The demo is **not** a clinical or personal emotion diagnosis tool.

## Project source and team credit

Original collaborative university project:
**[waysal1/Speech-Emotion-Recognition](https://github.com/waysal1/Speech-Emotion-Recognition)**.

- [Original team Jupyter notebook](https://github.com/waysal1/Speech-Emotion-Recognition/blob/main/CSE424_SER_Project_Code.ipynb)
- [Original team PDF report](https://github.com/waysal1/Speech-Emotion-Recognition/blob/main/CSE424%20Project%20Report.pdf)
- [Dataset preparation notes](notebooks/README.md)
- [New local audio-feature quickstart notebook](notebooks/audio_feature_quickstart.ipynb) (does not replace the original research notebook)

The original authors and team contributors retain credit. This portfolio
copy does not imply that Faraaz was the sole author of the coursework.

**Dataset citations:** [BanglaMood (CC BY 4.0)](https://data.mendeley.com/datasets/9cc8k5v7w4)
and [KUET Bangla Emotional Speech / KBES (CC BY 4.0)](https://data.mendeley.com/datasets/vsn37ps3rx).
Before using these for new training, check their current licenses and preserve
appropriate dataset attribution. Audio recordings are not shipped in the repo.

## Run the demo locally (Python 3.12)

```bash
python -m pip install -r demo/requirements.txt
streamlit run demo/app.py
```

## Deploy free on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io/ using GitHub.
2. Create a new app from `faraaz1027-cloud/Speech-Emotion-Recognition`.
3. Set branch **`portfolio-audio-demo`**, entrypoint **`demo/app.py`**.
4. Set Python version **3.12** under Advanced settings.
5. Deploy and try a short WAV file (less than 15 MB / 30 seconds).
6. Verify charts/feature-vector download, then add the *actual* app URL
   to this README and your CV; do not add an untested link.

The app uses dependencies from `demo/requirements.txt` as documented by
Streamlit. Its primary feature-extraction module is `demo/features.py`.

## Optional model reproduction (requires authorized audio)

If you have authorized audio with five label-named folders, this is a
**separate new training workflow**, not a replication of the original notebook:

```bash
python -m pip install -r training/requirements.txt
python training/train.py --data-dir /path/to/five-emotion-folders
```

Use `--manifest speakers.csv` (columns: `filename,speaker_id`) when speaker
metadata are available to produce a group-disjoint test split. Without such
metadata, it prints a warning about possible speaker leakage. The new workflow
fits scaling **inside the sklearn Pipeline on training records only**.

It can produce `models/ser_svm_pipeline.joblib` plus JSON evaluation
metadata after a successful local run. These outputs are ignored by Git
until reviewed, and **the Streamlit app does not use them yet**.

No new performance claims or production predictions are made before this
work and proper testing are complete.
