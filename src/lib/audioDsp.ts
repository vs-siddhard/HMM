/**
 * Audio Digital Signal Processing & MFCC Feature Extraction Engine
 * Provides pre-emphasis, framing, windowing, FFT, Mel-scale filterbanks,
 * Discrete Cosine Transform (DCT), and synthetic speech waveform generation.
 */

export interface MFCCConfig {
  sampleRate: number;      // e.g., 16000 Hz
  frameSizeMs: number;     // e.g., 25 ms
  frameStrideMs: number;   // e.g., 10 ms
  numFilterbanks: number;  // e.g., 26 mel filters
  numCepstral: number;     // e.g., 13 coefficients
  preEmphasis: number;     // e.g., 0.97
  lowFreqHz: number;       // e.g., 0 Hz
  highFreqHz: number;      // e.g., 8000 Hz (Nyquist)
}

export const DEFAULT_MFCC_CONFIG: MFCCConfig = {
  sampleRate: 16000,
  frameSizeMs: 25,
  frameStrideMs: 10,
  numFilterbanks: 26,
  numCepstral: 13,
  preEmphasis: 0.97,
  lowFreqHz: 0,
  highFreqHz: 8000,
};

export interface MFCCPipelineResult {
  rawSignal: Float32Array;
  preEmphasizedSignal: Float32Array;
  frames: Float32Array[];
  fftSpectrogram: number[][];    // [timeFrame][freqBin]
  melFilterbankEnergies: number[][]; // [timeFrame][filterIndex]
  mfccFeatures: number[][];     // [timeFrame][cepstralCoeff] (13 dims)
  deltaFeatures?: number[][];
  timeAxisMs: number[];
}

/**
 * Convert Hertz to Mel scale
 */
export function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

/**
 * Convert Mel scale to Hertz
 */
export function melToHz(mel: number): number {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

/**
 * Apply pre-emphasis filter: y[n] = x[n] - alpha * x[n-1]
 */
export function applyPreEmphasis(signal: Float32Array, alpha = 0.97): Float32Array {
  const result = new Float32Array(signal.length);
  result[0] = signal[0];
  for (let i = 1; i < signal.length; i++) {
    result[i] = signal[i] - alpha * signal[i - 1];
  }
  return result;
}

/**
 * Hamming window: w[n] = 0.54 - 0.46 * cos(2*pi*n / (N - 1))
 */
export function hammingWindow(N: number): Float32Array {
  const window = new Float32Array(N);
  for (let n = 0; n < N; n++) {
    window[n] = 0.54 - 0.46 * Math.cos((2 * Math.PI * n) / (N - 1));
  }
  return window;
}

/**
 * Simple Radix-2 Real FFT (magnitude spectrum)
 */
export function computeFFTMagnitude(frame: Float32Array, nFft = 512): number[] {
  // Zero-pad to nFft
  const real = new Float32Array(nFft);
  const imag = new Float32Array(nFft);
  for (let i = 0; i < Math.min(frame.length, nFft); i++) {
    real[i] = frame[i];
  }

  // Bit reversal permutation
  let j = 0;
  for (let i = 0; i < nFft - 1; i++) {
    if (i < j) {
      const tempR = real[i];
      real[i] = real[j];
      real[j] = tempR;
      const tempI = imag[i];
      imag[i] = imag[j];
      imag[j] = tempI;
    }
    let k = nFft >> 1;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }

  // Cooley-Tukey Radix-2 computation
  for (let len = 2; len <= nFft; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wStepR = Math.cos(angle);
    const wStepI = Math.sin(angle);

    for (let i = 0; i < nFft; i += len) {
      let wR = 1.0;
      let wI = 0.0;

      for (let k = 0; k < halfLen; k++) {
        const uR = real[i + k];
        const uI = imag[i + k];
        const vR = real[i + k + halfLen] * wR - imag[i + k + halfLen] * wI;
        const vI = real[i + k + halfLen] * wI + imag[i + k + halfLen] * wR;

        real[i + k] = uR + vR;
        imag[i + k] = uI + vI;
        real[i + k + halfLen] = uR - vR;
        imag[i + k + halfLen] = uI - vI;

        const nextWR = wR * wStepR - wI * wStepI;
        const nextWI = wR * wStepI + wI * wStepR;
        wR = nextWR;
        wI = nextWI;
      }
    }
  }

  // Half spectrum (Nyquist bins: nFft / 2 + 1)
  const numBins = Math.floor(nFft / 2) + 1;
  const magnitudes = new Array<number>(numBins);
  for (let i = 0; i < numBins; i++) {
    magnitudes[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]) / Math.sqrt(nFft);
  }
  return magnitudes;
}

/**
 * Construct triangular Mel filterbank weights
 */
export function createMelFilterbank(
  numFilters: number,
  nFft: number,
  sampleRate: number,
  lowFreq = 0,
  highFreq = sampleRate / 2
): number[][] {
  const lowMel = hzToMel(lowFreq);
  const highMel = hzToMel(highFreq);
  const melPoints = new Float32Array(numFilters + 2);
  const deltaMel = (highMel - lowMel) / (numFilters + 1);

  for (let i = 0; i < numFilters + 2; i++) {
    melPoints[i] = lowMel + i * deltaMel;
  }

  // Convert mel points to FFT bin indices
  const binIndices = new Int32Array(numFilters + 2);
  for (let i = 0; i < numFilters + 2; i++) {
    const hz = melToHz(melPoints[i]);
    binIndices[i] = Math.floor(((nFft + 1) * hz) / sampleRate);
  }

  const numBins = Math.floor(nFft / 2) + 1;
  const filterbank: number[][] = [];

  for (let m = 1; m <= numFilters; m++) {
    const filter = new Array<number>(numBins).fill(0);
    const left = binIndices[m - 1];
    const center = binIndices[m];
    const right = binIndices[m + 1];

    for (let k = left; k < center; k++) {
      if (center > left && k < numBins) {
        filter[k] = (k - left) / (center - left);
      }
    }
    for (let k = center; k <= right; k++) {
      if (right > center && k < numBins) {
        filter[k] = (right - k) / (right - center);
      }
    }
    filterbank.push(filter);
  }

  return filterbank;
}

/**
 * Discrete Cosine Transform Type-II (DCT-II)
 */
export function computeDCT(filterEnergies: number[], numCepstral = 13): number[] {
  const N = filterEnergies.length;
  const cepstral = new Array<number>(numCepstral).fill(0);

  for (let k = 0; k < numCepstral; k++) {
    let sum = 0;
    for (let n = 0; n < N; n++) {
      sum += filterEnergies[n] * Math.cos((Math.PI * k * (2 * n + 1)) / (2 * N));
    }
    // Standard orthonormal scaling
    const factor = k === 0 ? Math.sqrt(1 / N) : Math.sqrt(2 / N);
    cepstral[k] = sum * factor;
  }

  return cepstral;
}

/**
 * Compute first-order Delta features (velocity)
 */
export function computeDeltas(features: number[][], N = 2): number[][] {
  const numFrames = features.length;
  if (numFrames === 0) return [];
  const numFeatures = features[0].length;
  const deltas: number[][] = [];

  const denom = 2 * Array.from({ length: N }, (_, i) => Math.pow(i + 1, 2)).reduce((a, b) => a + b, 0);

  for (let t = 0; t < numFrames; t++) {
    const deltaVec = new Array<number>(numFeatures).fill(0);
    for (let d = 0; d < numFeatures; d++) {
      let num = 0;
      for (let n = 1; n <= N; n++) {
        const tPlus = Math.min(numFrames - 1, t + n);
        const tMinus = Math.max(0, t - n);
        num += n * (features[tPlus][d] - features[tMinus][d]);
      }
      deltaVec[d] = num / denom;
    }
    deltas.push(deltaVec);
  }

  return deltas;
}

/**
 * Full Pipeline: Raw Audio Signal -> Windowed Frames -> FFT Spectrogram -> Mel Filterbank -> MFCCs
 */
export function extractMFCCPipeline(
  signal: Float32Array,
  config: MFCCConfig = DEFAULT_MFCC_CONFIG
): MFCCPipelineResult {
  const frameLength = Math.floor((config.frameSizeMs / 1000) * config.sampleRate); // e.g. 400
  const frameStep = Math.floor((config.frameStrideMs / 1000) * config.sampleRate);   // e.g. 160
  const nFft = 512;

  // 1. Pre-emphasis
  const preEmphasized = applyPreEmphasis(signal, config.preEmphasis);

  // 2. Framing & Windowing
  const numFrames = Math.max(1, Math.floor((preEmphasized.length - frameLength) / frameStep) + 1);
  const window = hammingWindow(frameLength);
  const frames: Float32Array[] = [];
  const timeAxisMs: number[] = [];

  for (let i = 0; i < numFrames; i++) {
    const start = i * frameStep;
    const frame = new Float32Array(frameLength);
    for (let j = 0; j < frameLength; j++) {
      if (start + j < preEmphasized.length) {
        frame[j] = preEmphasized[start + j] * window[j];
      }
    }
    frames.push(frame);
    timeAxisMs.push((start / config.sampleRate) * 1000);
  }

  // 3. FFT Power Spectrum
  const fftSpectrogram: number[][] = [];
  for (let i = 0; i < frames.length; i++) {
    const mag = computeFFTMagnitude(frames[i], nFft);
    fftSpectrogram.push(mag);
  }

  // 4. Mel Filterbank
  const filterbank = createMelFilterbank(
    config.numFilterbanks,
    nFft,
    config.sampleRate,
    config.lowFreqHz,
    config.highFreqHz
  );

  const melFilterbankEnergies: number[][] = [];
  const mfccFeatures: number[][] = [];

  for (let i = 0; i < fftSpectrogram.length; i++) {
    const frameMag = fftSpectrogram[i];
    const energies: number[] = [];

    for (let m = 0; m < filterbank.length; m++) {
      let energy = 0;
      const weights = filterbank[m];
      for (let k = 0; k < frameMag.length; k++) {
        // power spectrum: |X|^2 / nFft
        const power = (frameMag[k] * frameMag[k]) / nFft;
        energy += power * weights[k];
      }
      // Log energy with numerical floor
      energies.push(Math.log(Math.max(energy, 1e-10)));
    }
    melFilterbankEnergies.push(energies);

    // 5. DCT to get 13 cepstral coefficients
    const cepstral = computeDCT(energies, config.numCepstral);
    mfccFeatures.push(cepstral);
  }

  const deltaFeatures = computeDeltas(mfccFeatures);

  return {
    rawSignal: signal,
    preEmphasizedSignal: preEmphasized,
    frames,
    fftSpectrogram,
    melFilterbankEnergies,
    mfccFeatures,
    deltaFeatures,
    timeAxisMs,
  };
}

/**
 * Generate synthetic realistic speech audio waveform for acoustic recognition
 * Each word has distinct acoustic formants, phoneme duration segments, and harmonic structures.
 */
export function generateSyntheticWordAudio(
  word: string,
  sampleRate = 16000,
  durationSec = 0.65,
  variation = 0
): Float32Array {
  const totalSamples = Math.floor(sampleRate * durationSec);
  const audio = new Float32Array(totalSamples);
  const normalizedWord = word.trim().toLowerCase();

  // Define formant profiles [F1, F2, F3, bandwidth, noiseRatio] for phoneme segments
  interface Segment {
    ratio: number; // fraction of total duration
    f1: number;
    f2: number;
    f3: number;
    f0: number;    // pitch fundamental
    noise: number; // aspiration / frication noise level
    amplitude: number;
  }

  let segments: Segment[] = [];

  // Random variation factor for pitch and formants
  const pitchJitter = 1 + (variation % 3 - 1) * 0.04;
  const formantShift = 1 + (variation % 2 === 0 ? 0.03 : -0.03);

  if (normalizedWord === 'apple') {
    segments = [
      // /æ/ vowel in "ap"
      { ratio: 0.38, f1: 720 * formantShift, f2: 1750 * formantShift, f3: 2500, f0: 130 * pitchJitter, noise: 0.05, amplitude: 0.8 },
      // /p/ closure & burst
      { ratio: 0.22, f1: 300, f2: 800, f3: 2000, f0: 0, noise: 0.45, amplitude: 0.3 },
      // /l/ lateral liquid
      { ratio: 0.40, f1: 420 * formantShift, f2: 1100 * formantShift, f3: 2800, f0: 125 * pitchJitter, noise: 0.03, amplitude: 0.7 },
    ];
  } else if (normalizedWord === 'banana') {
    segments = [
      // /b/ voiced stop
      { ratio: 0.15, f1: 250, f2: 850, f3: 2100, f0: 110 * pitchJitter, noise: 0.08, amplitude: 0.5 },
      // /æ/ vowel "ba"
      { ratio: 0.25, f1: 680 * formantShift, f2: 1650 * formantShift, f3: 2450, f0: 135 * pitchJitter, noise: 0.04, amplitude: 0.85 },
      // /n/ nasal "na"
      { ratio: 0.20, f1: 320, f2: 1350, f3: 2600, f0: 125 * pitchJitter, noise: 0.06, amplitude: 0.6 },
      // /ə/ vowel "na"
      { ratio: 0.25, f1: 520 * formantShift, f2: 1400 * formantShift, f3: 2400, f0: 120 * pitchJitter, noise: 0.03, amplitude: 0.75 },
      // ending fade
      { ratio: 0.15, f1: 450, f2: 1200, f3: 2200, f0: 110 * pitchJitter, noise: 0.04, amplitude: 0.4 },
    ];
  } else if (normalizedWord === 'cherry') {
    segments = [
      // /tʃ/ affricate (high frequency turbulence)
      { ratio: 0.30, f1: 400, f2: 2100, f3: 3600 * formantShift, f0: 0, noise: 0.85, amplitude: 0.7 },
      // /ɛ/ vowel "che"
      { ratio: 0.30, f1: 550 * formantShift, f2: 1850 * formantShift, f3: 2600, f0: 140 * pitchJitter, noise: 0.05, amplitude: 0.85 },
      // /r/ retroflex
      { ratio: 0.20, f1: 350, f2: 1300, f3: 1600 * formantShift, f0: 130 * pitchJitter, noise: 0.04, amplitude: 0.65 },
      // /i/ high front vowel "ry"
      { ratio: 0.20, f1: 280 * formantShift, f2: 2350 * formantShift, f3: 3000, f0: 145 * pitchJitter, noise: 0.02, amplitude: 0.8 },
    ];
  } else if (normalizedWord === 'yes') {
    segments = [
      // /j/ glide
      { ratio: 0.25, f1: 300, f2: 2200 * formantShift, f3: 3100, f0: 130 * pitchJitter, noise: 0.03, amplitude: 0.6 },
      // /ɛ/ vowel
      { ratio: 0.40, f1: 580 * formantShift, f2: 1800 * formantShift, f3: 2550, f0: 140 * pitchJitter, noise: 0.04, amplitude: 0.85 },
      // /s/ voiceless alveolar fricative
      { ratio: 0.35, f1: 300, f2: 2500, f3: 4500 * formantShift, f0: 0, noise: 0.9, amplitude: 0.75 },
    ];
  } else if (normalizedWord === 'no') {
    segments = [
      // /n/ nasal
      { ratio: 0.35, f1: 320, f2: 1350, f3: 2600, f0: 120 * pitchJitter, noise: 0.05, amplitude: 0.65 },
      // /oʊ/ diphthong
      { ratio: 0.65, f1: 500 * formantShift, f2: 950 * formantShift, f3: 2300, f0: 125 * pitchJitter, noise: 0.03, amplitude: 0.85 },
    ];
  } else {
    // Generic fallback based on word hash
    const hash = normalizedWord.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0);
    const baseFreq = 400 + (hash % 10) * 80;
    segments = [
      { ratio: 0.33, f1: baseFreq, f2: baseFreq * 2.2, f3: baseFreq * 3.5, f0: 130, noise: 0.1, amplitude: 0.7 },
      { ratio: 0.34, f1: baseFreq * 1.3, f2: baseFreq * 2.6, f3: baseFreq * 4.0, f0: 135, noise: 0.05, amplitude: 0.85 },
      { ratio: 0.33, f1: baseFreq * 0.9, f2: baseFreq * 1.8, f3: baseFreq * 3.2, f0: 120, noise: 0.08, amplitude: 0.6 },
    ];
  }

  // Synthesize acoustic signal by interpolating segments with Klatt-like formant filtering
  let sampleIndex = 0;
  let phaseF0 = 0;
  let prevSample = 0;

  for (let s = 0; s < segments.length; s++) {
    const seg = segments[s];
    const segSamples = Math.floor(seg.ratio * totalSamples);
    const endSample = Math.min(totalSamples, sampleIndex + segSamples);

    for (let i = sampleIndex; i < endSample; i++) {
      const t = i / sampleRate;
      let val = 0;

      // Voiced glottal pulse train
      if (seg.f0 > 0) {
        phaseF0 += (2 * Math.PI * seg.f0) / sampleRate;
        if (phaseF0 > 2 * Math.PI) phaseF0 -= 2 * Math.PI;

        // Harmonic glottal pulse (sum of harmonics)
        const glottal =
          Math.sin(phaseF0) +
          0.6 * Math.sin(2 * phaseF0) +
          0.35 * Math.sin(3 * phaseF0) +
          0.2 * Math.sin(4 * phaseF0);

        // Resonant formant resonators
        const r1 = Math.sin(2 * Math.PI * seg.f1 * t) * Math.exp(-((t * 20) % 1) * 3);
        const r2 = 0.5 * Math.sin(2 * Math.PI * seg.f2 * t) * Math.exp(-((t * 20) % 1) * 4);
        const r3 = 0.25 * Math.sin(2 * Math.PI * seg.f3 * t);

        val = (glottal * 0.4 + r1 * 0.35 + r2 * 0.2 + r3 * 0.05) * (1 - seg.noise);
      }

      // Unvoiced frication / turbulence noise
      if (seg.noise > 0) {
        const whiteNoise = (Math.random() * 2 - 1);
        // Simple bandpass emulation for high-frequency noise
        const filteredNoise = whiteNoise - 0.7 * prevSample;
        val += filteredNoise * seg.noise * 0.8;
      }

      // Smooth envelope attack and decay
      const envProgress = i / totalSamples;
      const attack = Math.min(1, i / (0.04 * sampleRate));
      const release = Math.min(1, (totalSamples - i) / (0.06 * sampleRate));
      const envelope = attack * release;

      const currentSample = val * seg.amplitude * envelope;
      audio[i] = currentSample;
      prevSample = currentSample;
    }
    sampleIndex = endSample;
  }

  // Normalize audio to -1.0 to 1.0 with 0.85 peak
  let maxAbs = 0;
  for (let i = 0; i < audio.length; i++) {
    const a = Math.abs(audio[i]);
    if (a > maxAbs) maxAbs = a;
  }
  if (maxAbs > 0) {
    const scale = 0.85 / maxAbs;
    for (let i = 0; i < audio.length; i++) {
      audio[i] *= scale;
    }
  }

  return audio;
}

/**
 * Encode Float32Array into a standard 16-bit PCM WAV Blob
 */
export function float32ArrayToWavBlob(samples: Float32Array, sampleRate = 16000): Blob {
  const numChannels = 1;
  const bytesPerSample = 2; // 16-bit
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // RIFF identifier
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');

  // fmt chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true);  // AudioFormat (1 = PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true); // BitsPerSample

  // data chunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // Write PCM samples
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    const intSample = s < 0 ? s * 0x8000 : s * 0x7fff;
    view.setInt16(offset, intSample, true);
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

function writeString(view: DataView, offset: number, string: string): void {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}
