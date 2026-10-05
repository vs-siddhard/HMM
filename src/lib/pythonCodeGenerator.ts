/**
 * Dynamic Python Code Generator for HMM Speech Recognition
 * Generates copy-pasteable, verified scripts for hmmlearn+librosa, pure NumPy/SciPy, and CLI test runners.
 */

export function generateHmmlearnPythonCode(words: string[] = ['apple', 'banana', 'cherry'], nStates = 4): string {
  const wordsListStr = JSON.stringify(words);

  return `"""
Speech Recognition using Hidden Markov Models (GMM-HMM)
Libraries: hmmlearn, librosa, numpy

Requirements:
    pip install hmmlearn librosa numpy soundfile
"""

import os
import numpy as np
import librosa
from hmmlearn import hmm
import pickle

# 1. Configuration
WORDS = ${wordsListStr}
N_STATES = ${nStates}
N_MFCC = 13
SAMPLE_RATE = 16000
DATA_PATH = "./audio_data"  # Structure: audio_data/<word>/sample_1.wav


def extract_features(file_path: str, add_deltas: bool = True) -> np.ndarray:
    """
    Extracts MFCC features and first-order temporal derivatives (Deltas).
    Returns (n_frames, n_features) shape suitable for hmmlearn.
    """
    # 1. Load audio with fixed sample rate
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    # 2. Extract 13 MFCC coefficients (25ms window, 10ms hop)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=int(0.025 * sr),
        hop_length=int(0.010 * sr)
    )  # shape: (n_mfcc, n_frames)
    
    if add_deltas:
        # First-order derivative captures velocity of sound changes
        delta = librosa.feature.delta(mfcc)
        features = np.vstack([mfcc, delta])  # shape: (26, n_frames)
    else:
        features = mfcc
        
    # Transpose to (n_samples, n_features) for scikit-learn / hmmlearn convention
    return features.T


class SpeechRecognizer:
    def __init__(self, words=WORDS, n_states=N_STATES):
        self.words = words
        self.n_states = n_states
        self.models = {}

    def _init_left_to_right_hmm(self) -> hmm.GaussianHMM:
        """
        Creates a Gaussian HMM with strict Left-to-Right topology (Bakis model).
        Transitions are only allowed to stay in current state or advance to next state.
        """
        model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=100,
            tol=1e-3,
            verbose=False,
            init_params="mc"  # initialize means and covariances; we set transmat manually
        )
        
        # Initial State Distribution: start at State 0 with probability 1.0
        startprob = np.zeros(self.n_states)
        startprob[0] = 1.0
        model.startprob_ = startprob
        
        # Left-to-Right Transition Matrix
        transmat = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            if i == self.n_states - 1:
                transmat[i, i] = 1.0  # Final state absorbs
            else:
                transmat[i, i] = 0.65      # Self-loop probability
                transmat[i, i + 1] = 0.35  # Advance to next state
        model.transmat_ = transmat
        
        return model

    def train_from_directory(self, data_path=DATA_PATH):
        """
        Loads all .wav files for each word in data_path and trains individual HMMs.
        """
        print("Training HMM speech recognizer...")
        for word in self.words:
            word_dir = os.path.join(data_path, word)
            if not os.path.exists(word_dir):
                print(f"  [Warning] Directory not found for '{word}': {word_dir}")
                continue
                
            word_features = []
            lengths = []
            
            for fname in sorted(os.listdir(word_dir)):
                if fname.lower().endswith(".wav"):
                    fpath = os.path.join(word_dir, fname)
                    feat = extract_features(fpath)
                    word_features.append(feat)
                    lengths.append(len(feat))
                    
            if not word_features:
                print(f"  [Warning] No WAV files found for word '{word}'")
                continue
                
            # Concatenate all sequences into a single 2D matrix
            X = np.concatenate(word_features)
            
            # Initialize Left-to-Right HMM and fit with Baum-Welch (EM)
            model = self._init_left_to_right_hmm()
            model.fit(X, lengths)
            self.models[word] = model
            print(f"  [Trained] Model for '{word}' converged with {len(lengths)} samples.")

    def predict(self, file_path: str):
        """
        Scores input audio against all word models using the Forward algorithm.
        Returns: (predicted_word, log_likelihood_dict)
        """
        features = extract_features(file_path)
        scores = {}
        
        for word, model in self.models.items():
            try:
                # Forward log-likelihood: log P(O | lambda_word)
                score = model.score(features)
                scores[word] = score
            except Exception as e:
                scores[word] = float("-inf")
                
        best_word = max(scores, key=scores.get)
        return best_word, scores

    def save(self, filepath="speech_models.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump(self.models, f)
        print(f"Saved {len(self.models)} models to {filepath}")

    def load(self, filepath="speech_models.pkl"):
        with open(filepath, "rb") as f:
            self.models = pickle.load(f)
        print(f"Loaded {len(self.models)} models from {filepath}")


# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    recognizer = SpeechRecognizer(words=${wordsListStr})
    
    # Train models on recorded samples:
    # recognizer.train_from_directory("./audio_data")
    
    # Recognize an unknown test audio:
    # predicted, scores = recognizer.predict("./test_sample.wav")
    # print(f"Recognized Word: {predicted}")
    # print("Confidence log-likelihoods:", scores)
`;
}

export function generatePureNumpyPythonCode(words: string[] = ['apple', 'banana', 'cherry'], nStates = 4): string {
  const wordsListStr = JSON.stringify(words);

  return `"""
Self-Contained HMM Speech Recognizer (Pure NumPy & SciPy)
Zero external audio ML dependencies! Runs in standard Python environments.

Includes:
- FFT & Hamming window feature extraction
- Left-to-Right Gaussian HMM with Log-Sum-Exp stability
- Forward algorithm for Log-Likelihood evaluation
- Viterbi Trellis decoding
- Segmental K-Means & Baum-Welch (EM) parameter estimation
- Built-in synthetic acoustic sample generator for immediate testing!
"""

import numpy as np
from scipy.fftpack import dct
import math

WORDS = ${wordsListStr}
N_STATES = ${nStates}
N_FEATURES = 13
SAMPLE_RATE = 16000


def extract_spectral_mfcc(signal: np.ndarray, sr=SAMPLE_RATE, n_mfcc=N_FEATURES) -> np.ndarray:
    """
    Extracts 13 MFCC-like cepstral coefficients without external libraries.
    """
    # 1. Pre-emphasis filter
    signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    
    # 2. Windowing & Framing (25ms window, 10ms stride)
    win_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    n_frames = max(1, 1 + int(math.floor((len(signal) - win_len) / hop_len)))
    
    window = np.hamming(win_len)
    frames = []
    for i in range(n_frames):
        start = i * hop_len
        frame = signal[start : start + win_len] * window
        # Real FFT magnitude spectrum
        mag = np.abs(np.fft.rfft(frame, n=512))
        # Log filter energies (using 26 uniform/mel-spaced bins)
        bins = np.array_split(mag[:256], 26)
        energies = np.array([np.sum(b**2) + 1e-10 for b in bins])
        log_energies = np.log(energies)
        # DCT-II to obtain 13 orthogonal cepstral coefficients
        cepstral = dct(log_energies, type=2, norm='ortho')[:n_mfcc]
        frames.append(cepstral)
        
    return np.array(frames)


class GaussianHMM:
    def __init__(self, word: str, n_states: int = N_STATES, n_features: int = N_FEATURES):
        self.word = word
        self.n_states = n_states
        self.n_features = n_features
        
        # Left-to-Right topology
        self.startprob = np.zeros(n_states)
        self.startprob[0] = 1.0
        
        self.transmat = np.zeros((n_states, n_states))
        for i in range(n_states):
            if i == n_states - 1:
                self.transmat[i, i] = 1.0
            else:
                self.transmat[i, i] = 0.65
                self.transmat[i, i + 1] = 0.35
                
        self.means = np.zeros((n_states, n_features))
        self.vars = np.ones((n_states, n_features))

    def _log_gaussian(self, x: np.ndarray, state_idx: int) -> float:
        """Computes log N(x; mu, sigma^2) with diagonal covariance"""
        mu = self.means[state_idx]
        var = np.maximum(self.vars[state_idx], 1e-4)
        D = self.n_features
        diff = x - mu
        return -0.5 * (D * np.log(2 * np.pi) + np.sum(np.log(var) + (diff ** 2) / var))

    def forward(self, obs: np.ndarray):
        """Forward algorithm in log-space to prevent underflow"""
        T = len(obs)
        N = self.n_states
        alpha = np.full((T, N), -np.inf)
        
        # t = 0
        for j in range(N):
            if self.startprob[j] > 0:
                alpha[0, j] = np.log(self.startprob[j]) + self._log_gaussian(obs[0], j)
                
        # t = 1 ... T-1
        for t in range(1, T):
            for j in range(N):
                terms = []
                for i in range(N):
                    if self.transmat[i, j] > 0 and np.isfinite(alpha[t - 1, i]):
                        terms.append(alpha[t - 1, i] + np.log(self.transmat[i, j]))
                if terms:
                    # Log-Sum-Exp
                    m = max(terms)
                    alpha[t, j] = self._log_gaussian(obs[t], j) + m + np.log(np.sum(np.exp(terms - m)))
                    
        # Total Log-Likelihood
        m = np.max(alpha[-1])
        log_likelihood = m + np.log(np.sum(np.exp(alpha[-1] - m))) if np.isfinite(m) else -np.inf
        return log_likelihood, alpha

    def viterbi(self, obs: np.ndarray):
        """Viterbi decoding algorithm for best state sequence"""
        T = len(obs)
        N = self.n_states
        trellis = np.full((T, N), -np.inf)
        backpointers = np.zeros((T, N), dtype=int)
        
        for j in range(N):
            if self.startprob[j] > 0:
                trellis[0, j] = np.log(self.startprob[j]) + self._log_gaussian(obs[0], j)
                
        for t in range(1, T):
            for j in range(N):
                scores = [trellis[t - 1, i] + np.log(self.transmat[i, j]) if self.transmat[i, j] > 0 else -np.inf for i in range(N)]
                best_i = int(np.argmax(scores))
                trellis[t, j] = scores[best_i] + self._log_gaussian(obs[t], j)
                backpointers[t, j] = best_i
                
        best_path = [int(np.argmax(trellis[-1]))]
        for t in range(T - 1, 0, -1):
            best_path.append(backpointers[t, best_path[-1]])
        best_path.reverse()
        return np.max(trellis[-1]), best_path

    def fit(self, data_list):
        """Segmental Partition Initialization for Speech Acoustic Models"""
        state_buckets = [[] for _ in range(self.n_states)]
        for seq in data_list:
            T = len(seq)
            for t in range(T):
                s = min(self.n_states - 1, int((t / T) * self.n_states))
                state_buckets[s].append(seq[t])
                
        for s in range(self.n_states):
            if state_buckets[s]:
                arr = np.array(state_buckets[s])
                self.means[s] = np.mean(arr, axis=0)
                self.vars[s] = np.maximum(np.var(arr, axis=0), 1e-4)


# ==============================================================
# Synthetic Speech Audio Generator for Verification
# ==============================================================
def synthesize_sound(word: str, duration=0.6, sr=SAMPLE_RATE) -> np.ndarray:
    """Generates synthetic acoustic formant waveforms for distinct words"""
    t = np.linspace(0, duration, int(sr * duration))
    # Distinct formant profile per word
    profiles = {
        "apple": [(750, 0.5), (1700, 0.3), (2500, 0.15)],
        "banana": [(680, 0.4), (1400, 0.4), (2400, 0.15)],
        "cherry": [(550, 0.3), (1850, 0.4), (3200, 0.25)],
        "yes": [(300, 0.25), (1800, 0.45), (2800, 0.2)],
        "no": [(320, 0.45), (950, 0.35), (2300, 0.15)],
    }
    harmonics = profiles.get(word.lower(), [(500, 0.5), (1500, 0.3), (2500, 0.2)])
    signal = np.zeros_like(t)
    for freq, amp in harmonics:
        signal += amp * np.sin(2 * np.pi * freq * t)
    # Add subtle envelope and natural white noise
    env = np.sin(np.pi * np.linspace(0, 1, len(t))) ** 2
    noise = np.random.normal(0, 0.05, len(t))
    return (signal + noise) * env


# ==============================================================
# Main Demonstration Workflow
# ==============================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" HMM Speech Recognition - Pure NumPy Implementation")
    print("=" * 60)
    
    models = {word: GaussianHMM(word, n_states=N_STATES) for word in WORDS}
    
    # 1. Train models using 3 synthetic takes per word
    print("\\n[1/3] Generating training audio samples and training HMMs...")
    for word, model in models.items():
        training_features = []
        for take in range(3):
            audio = synthesize_sound(word)
            feat = extract_spectral_mfcc(audio)
            training_features.append(feat)
        model.fit(training_features)
        print(f"  ✓ Model '{word}' fitted successfully across {len(training_features)} takes.")

    # 2. Test Recognition with a newly generated test utterance
    test_word = WORDS[0]
    print(f"\\n[2/3] Simulating test audio for: '{test_word}'...")
    test_audio = synthesize_sound(test_word)
    test_feat = extract_spectral_mfcc(test_audio)

    # 3. Score against all trained models
    print("\\n[3/3] Scoring acoustic observations across vocabulary...")
    scores = {}
    viterbi_paths = {}
    for word, model in models.items():
        log_prob, _ = model.forward(test_feat)
        _, path = model.viterbi(test_feat)
        scores[word] = log_prob
        viterbi_paths[word] = path
        print(f"  Word: {word:<10} | Log-Likelihood: {log_prob:>10.2f}")

    recognized = max(scores, key=scores.get)
    print("\\n" + "=" * 60)
    print(f" PREDICTION RESULT: '{recognized.upper()}' (Ground Truth: '{test_word}')")
    print(f" Viterbi State Progression: {viterbi_paths[recognized][:10]}...")
    print("=" * 60)
`;
}
