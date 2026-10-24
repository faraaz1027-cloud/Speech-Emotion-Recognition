import io
import unittest

import numpy as np
import soundfile as sf
from demo.features import extract_features, load_audio


class AudioFeaturesTests(unittest.TestCase):
    @staticmethod
    def synth_wav(duration=1.2, sample_rate=22050):
        t = np.arange(int(sample_rate * duration)) / sample_rate
        samples = (0.22 * np.sin(2.0 * np.pi * 220 * t)).astype(np.float32)
        out = io.BytesIO()
        sf.write(out, samples, sample_rate, format="WAV")
        return out.getvalue()

    def test_58_finite_features(self):
        audio, sr = load_audio(self.synth_wav())
        features = extract_features(audio, sr)
        self.assertEqual(len(features), 58)
        self.assertTrue(np.all(np.isfinite(list(features.values()))))
        self.assertIn("mfcc_20_std", features)
        self.assertIn("chroma_12_mean", features)
        self.assertIn("tempo", features)

    def test_empty_audio_is_rejected(self):
        with self.assertRaises(ValueError):
            load_audio(b"")

    def test_too_long_is_rejected(self):
        with self.assertRaises(ValueError):
            load_audio(self.synth_wav(duration=31.0))

    def test_silence_is_rejected(self):
        data = io.BytesIO()
        sf.write(data, np.zeros(22050), 22050, format="WAV")
        with self.assertRaises(ValueError):
            load_audio(data.getvalue())


if __name__ == "__main__":
    unittest.main()
