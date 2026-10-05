import { extractMFCCPipeline, generateSyntheticWordAudio, float32ArrayToWavBlob, MFCCPipelineResult } from './audioDsp';
import { SpeechRecognizer } from './hmmEngine';

export interface AudioSampleItem {
  id: string;
  word: string;
  label: string;
  source: 'preset' | 'recorded' | 'uploaded';
  durationSec: number;
  sampleRate: number;
  rawSignal: Float32Array;
  wavBlob?: Blob;
  pipelineResult?: MFCCPipelineResult;
  timestamp: number;
}

export const DEFAULT_VOCABULARY = ['apple', 'banana', 'cherry', 'yes', 'no'];

/**
 * Generate standard preset audio samples and extract their MFCC feature sequences
 */
export function buildPresetDataset(): {
  samples: AudioSampleItem[];
  recognizer: SpeechRecognizer;
} {
  const samples: AudioSampleItem[] = [];
  const recognizer = new SpeechRecognizer(DEFAULT_VOCABULARY, 4, 13);

  for (const word of DEFAULT_VOCABULARY) {
    const wordFeatures: number[][][] = [];

    // Create 3 acoustic variations per word to train the HMM
    for (let varIdx = 0; varIdx < 3; varIdx++) {
      const duration = 0.55 + varIdx * 0.06;
      const rawSignal = generateSyntheticWordAudio(word, 16000, duration, varIdx);
      const wavBlob = float32ArrayToWavBlob(rawSignal, 16000);
      const pipelineResult = extractMFCCPipeline(rawSignal);

      const sampleItem: AudioSampleItem = {
        id: `${word}_sample_${varIdx + 1}`,
        word,
        label: `${word.toUpperCase()} · Take ${varIdx + 1}`,
        source: 'preset',
        durationSec: duration,
        sampleRate: 16000,
        rawSignal,
        wavBlob,
        pipelineResult,
        timestamp: Date.now() - (3 - varIdx) * 10000,
      };

      samples.push(sampleItem);
      wordFeatures.push(pipelineResult.mfccFeatures);
    }

    // Train the HMM with Baum-Welch (EM)
    recognizer.trainWord(word, wordFeatures, true, 8);
  }

  return { samples, recognizer };
}
