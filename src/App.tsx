/**
 * AcousticHMM — Speech Recognition & Hidden Markov Model Workbench
 * @license Apache-2.0
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Header } from './components/Header';
import { AudioInputControls } from './components/AudioInputControls';
import { ClassifierView } from './components/ClassifierView';
import { HMMInspectorView } from './components/HMMInspectorView';
import { MFCCLabView } from './components/MFCCLabView';
import { TrainingLabView } from './components/TrainingLabView';
import { PythonLabView } from './components/PythonLabView';

import {
  AudioSampleItem,
  buildPresetDataset,
  DEFAULT_VOCABULARY,
} from './lib/presetDataset';
import {
  extractMFCCPipeline,
  generateSyntheticWordAudio,
  float32ArrayToWavBlob,
} from './lib/audioDsp';
import { RecognitionResult, SpeechRecognizer } from './lib/hmmEngine';

export default function App() {
  const [activeTab, setActiveTab] = useState<'classifier' | 'trellis' | 'mfcc' | 'training' | 'python'>('classifier');
  const [samples, setSamples] = useState<AudioSampleItem[]>([]);
  const [vocabulary, setVocabulary] = useState<string[]>(DEFAULT_VOCABULARY);
  const [recognizer, setRecognizer] = useState<SpeechRecognizer | null>(null);
  const [currentSample, setCurrentSample] = useState<AudioSampleItem | null>(null);
  const [recognitionResult, setRecognitionResult] = useState<RecognitionResult | null>(null);
  const [selectedInspectWord, setSelectedInspectWord] = useState<string>('apple');

  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [isRetraining, setIsRetraining] = useState<boolean>(false);
  const [isSimulating, setIsSimulating] = useState<boolean>(false);

  // Audio Context references
  const audioContextRef = useRef<AudioContext | null>(null);
  const activeAudioSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);

  // Initialize dataset and models on mount
  useEffect(() => {
    const { samples: initSamples, recognizer: initRecognizer } = buildPresetDataset();
    setSamples(initSamples);
    setRecognizer(initRecognizer);

    if (initSamples.length > 0) {
      const firstSample = initSamples[0];
      setCurrentSample(firstSample);

      if (firstSample.pipelineResult?.mfccFeatures) {
        const res = initRecognizer.predict(firstSample.pipelineResult.mfccFeatures);
        setRecognitionResult(res);
      }
    }
  }, []);

  // Update prediction when currentSample or recognizer changes
  const runRecognition = useCallback(
    (sample: AudioSampleItem, rec: SpeechRecognizer) => {
      if (!sample.pipelineResult?.mfccFeatures) {
        sample.pipelineResult = extractMFCCPipeline(sample.rawSignal);
      }
      const res = rec.predict(sample.pipelineResult.mfccFeatures);
      setRecognitionResult(res);
    },
    []
  );

  const handleSelectSample = (sample: AudioSampleItem) => {
    setCurrentSample(sample);
    if (recognizer) {
      runRecognition(sample, recognizer);
    }
  };

  // Play audio signal through Web Audio API
  const handlePlayAudio = () => {
    if (!currentSample?.rawSignal) return;

    try {
      if (!audioContextRef.current || audioContextRef.current.state === 'suspended') {
        const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        audioContextRef.current = new AudioCtx();
      }
      const ctx = audioContextRef.current;

      // Stop previous playing source if any
      if (activeAudioSourceRef.current) {
        try {
          activeAudioSourceRef.current.stop();
        } catch {
          // ignore
        }
      }

      const signal = currentSample.rawSignal;
      const buffer = ctx.createBuffer(1, signal.length, currentSample.sampleRate || 16000);
      buffer.getChannelData(0).set(signal);

      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);

      source.onended = () => {
        setIsPlaying(false);
      };

      setIsPlaying(true);
      activeAudioSourceRef.current = source;
      source.start();
    } catch (err) {
      console.error('Audio playback error:', err);
      setIsPlaying(false);
    }
  };

  // Quick Random Simulation
  const handleQuickSimulate = () => {
    if (samples.length === 0 || !recognizer) return;
    setIsSimulating(true);

    const randomIndex = Math.floor(Math.random() * samples.length);
    const chosenSample = samples[randomIndex];
    setCurrentSample(chosenSample);
    runRecognition(chosenSample, recognizer);

    setTimeout(() => {
      setIsSimulating(false);
    }, 250);
  };

  // Microphone Recording
  const handleStartRecord = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordedChunksRef.current = [];

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          recordedChunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const blob = new Blob(recordedChunksRef.current, { type: 'audio/webm' });
        const arrayBuffer = await blob.arrayBuffer();

        const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        const decodeCtx = new AudioCtx();
        const audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer);

        // Resample / downmix to 16kHz mono
        const rawChannel = audioBuffer.getChannelData(0);
        const targetRate = 16000;
        const targetLength = Math.floor((rawChannel.length * targetRate) / audioBuffer.sampleRate);
        const resampled = new Float32Array(targetLength);

        for (let i = 0; i < targetLength; i++) {
          const srcIdx = Math.floor((i * audioBuffer.sampleRate) / targetRate);
          resampled[i] = rawChannel[srcIdx] || 0;
        }

        const duration = resampled.length / targetRate;
        const pipelineResult = extractMFCCPipeline(resampled);

        const newSample: AudioSampleItem = {
          id: `mic_${Date.now()}`,
          word: vocabulary[0] || 'speech',
          label: `Microphone Recording · ${(duration * 1000).toFixed(0)}ms`,
          source: 'recorded',
          durationSec: duration,
          sampleRate: 16000,
          rawSignal: resampled,
          wavBlob: float32ArrayToWavBlob(resampled, 16000),
          pipelineResult,
          timestamp: Date.now(),
        };

        setSamples((prev) => [newSample, ...prev]);
        setCurrentSample(newSample);

        if (recognizer) {
          runRecognition(newSample, recognizer);
        }

        // Stop microphone stream tracks
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.warn('Microphone access denied or unavailable:', err);
      // Fallback: notify with synthetic recording simulation
      const fallbackSignal = generateSyntheticWordAudio(vocabulary[0] || 'apple', 16000, 0.7, 9);
      const pipelineResult = extractMFCCPipeline(fallbackSignal);
      const newSample: AudioSampleItem = {
        id: `mic_sim_${Date.now()}`,
        word: vocabulary[0] || 'apple',
        label: `Simulated Voice Capture · 700ms`,
        source: 'recorded',
        durationSec: 0.7,
        sampleRate: 16000,
        rawSignal: fallbackSignal,
        wavBlob: float32ArrayToWavBlob(fallbackSignal, 16000),
        pipelineResult,
        timestamp: Date.now(),
      };
      setSamples((prev) => [newSample, ...prev]);
      setCurrentSample(newSample);
      if (recognizer) runRecognition(newSample, recognizer);
      setIsRecording(false);
    }
  };

  const handleStopRecord = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  // Synthesize custom acoustic formant sample
  const handleSynthesizeCustom = (word: string, duration: number) => {
    const rawSignal = generateSyntheticWordAudio(word, 16000, duration, Math.floor(Math.random() * 10));
    const pipelineResult = extractMFCCPipeline(rawSignal);

    const newSample: AudioSampleItem = {
      id: `synth_${word}_${Date.now()}`,
      word: word.toLowerCase(),
      label: `Acoustic Synth · ${word.toUpperCase()} (${(duration * 1000).toFixed(0)}ms)`,
      source: 'preset',
      durationSec: duration,
      sampleRate: 16000,
      rawSignal,
      wavBlob: float32ArrayToWavBlob(rawSignal, 16000),
      pipelineResult,
      timestamp: Date.now(),
    };

    setSamples((prev) => [newSample, ...prev]);
    setCurrentSample(newSample);
    if (recognizer) {
      runRecognition(newSample, recognizer);
    }
  };

  // Upload Audio File
  const handleUploadAudio = async (file: File) => {
    try {
      const arrayBuffer = await file.arrayBuffer();
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const ctx = new AudioCtx();
      const audioBuffer = await ctx.decodeAudioData(arrayBuffer);

      const rawChannel = audioBuffer.getChannelData(0);
      const targetRate = 16000;
      const targetLength = Math.floor((rawChannel.length * targetRate) / audioBuffer.sampleRate);
      const resampled = new Float32Array(targetLength);

      for (let i = 0; i < targetLength; i++) {
        const srcIdx = Math.floor((i * audioBuffer.sampleRate) / targetRate);
        resampled[i] = rawChannel[srcIdx] || 0;
      }

      const duration = resampled.length / targetRate;
      const pipelineResult = extractMFCCPipeline(resampled);

      const cleanName = file.name.replace(/\.[^/.]+$/, '').toLowerCase();
      const detectedWord = vocabulary.find((w) => cleanName.includes(w)) || vocabulary[0];

      const newSample: AudioSampleItem = {
        id: `upload_${Date.now()}`,
        word: detectedWord,
        label: `Uploaded: ${file.name} (${(duration * 1000).toFixed(0)}ms)`,
        source: 'uploaded',
        durationSec: duration,
        sampleRate: 16000,
        rawSignal: resampled,
        wavBlob: float32ArrayToWavBlob(resampled, 16000),
        pipelineResult,
        timestamp: Date.now(),
      };

      setSamples((prev) => [newSample, ...prev]);
      setCurrentSample(newSample);
      if (recognizer) {
        runRecognition(newSample, recognizer);
      }
    } catch (err) {
      console.error('Audio upload decode error:', err);
    }
  };

  // Add new word to vocabulary
  const handleAddWord = (word: string) => {
    const normalized = word.trim().toLowerCase();
    if (!normalized || vocabulary.includes(normalized)) return;

    const newVocab = [...vocabulary, normalized];
    setVocabulary(newVocab);

    if (recognizer) {
      // Create 3 synthetic takes for the new word
      const newSamples: AudioSampleItem[] = [];
      const wordFeatures: number[][][] = [];

      for (let i = 0; i < 3; i++) {
        const duration = 0.55 + i * 0.05;
        const rawSignal = generateSyntheticWordAudio(normalized, 16000, duration, i);
        const pipelineResult = extractMFCCPipeline(rawSignal);

        const sampleItem: AudioSampleItem = {
          id: `${normalized}_sample_${i + 1}`,
          word: normalized,
          label: `${normalized.toUpperCase()} · Take ${i + 1}`,
          source: 'preset',
          durationSec: duration,
          sampleRate: 16000,
          rawSignal,
          wavBlob: float32ArrayToWavBlob(rawSignal, 16000),
          pipelineResult,
          timestamp: Date.now(),
        };

        newSamples.push(sampleItem);
        wordFeatures.push(pipelineResult.mfccFeatures);
      }

      recognizer.trainWord(normalized, wordFeatures, true, 8);
      setSamples((prev) => [...newSamples, ...prev]);

      if (newSamples.length > 0) {
        setCurrentSample(newSamples[0]);
        runRecognition(newSamples[0], recognizer);
      }
    }
  };

  // Re-train all models across all current samples
  const handleRetrainAll = () => {
    if (!recognizer) return;
    setIsRetraining(true);

    setTimeout(() => {
      for (const word of vocabulary) {
        const wordSamples = samples.filter((s) => s.word.toLowerCase() === word.toLowerCase());
        const dataList: number[][][] = [];
        for (const s of wordSamples) {
          if (s.pipelineResult?.mfccFeatures) {
            dataList.push(s.pipelineResult.mfccFeatures);
          }
        }
        if (dataList.length > 0) {
          recognizer.trainWord(word, dataList, true, 8);
        }
      }

      setIsRetraining(false);
      if (currentSample) {
        runRecognition(currentSample, recognizer);
      }
    }, 200);
  };

  const handleSelectWordToInspect = (word: string) => {
    setSelectedInspectWord(word);
    setActiveTab('trellis');
  };

  return (
    <div className="min-h-screen bg-black text-white flex flex-col font-sans selection:bg-neutral-800 selection:text-white">
      {/* Top Bar Navigation */}
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        isRecording={isRecording}
        onToggleRecord={isRecording ? handleStopRecord : handleStartRecord}
        onQuickSimulate={handleQuickSimulate}
        isSimulating={isSimulating}
      />

      {/* Main Workspace */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* Universal Audio Input Ribbon */}
        <AudioInputControls
          currentSample={currentSample}
          onSelectSample={handleSelectSample}
          samples={samples}
          vocabulary={vocabulary}
          isRecording={isRecording}
          onStartRecord={handleStartRecord}
          onStopRecord={handleStopRecord}
          onSynthesizeCustom={handleSynthesizeCustom}
          onUploadAudio={handleUploadAudio}
          onPlayAudio={handlePlayAudio}
          isPlaying={isPlaying}
        />

        {/* Dynamic Tab Panes */}
        {activeTab === 'classifier' && (
          <ClassifierView
            result={recognitionResult}
            currentSample={currentSample}
            vocabulary={vocabulary}
            samples={samples}
            onAddWord={handleAddWord}
            onRetrainAll={handleRetrainAll}
            isRetraining={isRetraining}
            onPlayAudio={handlePlayAudio}
            isPlaying={isPlaying}
            onSelectWordToInspect={handleSelectWordToInspect}
          />
        )}

        {activeTab === 'trellis' && (
          <HMMInspectorView
            models={recognizer?.models || new Map()}
            vocabulary={vocabulary}
            currentSample={currentSample}
            selectedWord={selectedInspectWord}
            onSelectWord={setSelectedInspectWord}
          />
        )}

        {activeTab === 'mfcc' && <MFCCLabView currentSample={currentSample} />}

        {activeTab === 'training' && recognizer && (
          <TrainingLabView
            recognizer={recognizer}
            samples={samples}
            vocabulary={vocabulary}
            onModelRetrained={() => {
              if (currentSample) {
                runRecognition(currentSample, recognizer);
              }
            }}
          />
        )}

        {activeTab === 'python' && (
          <PythonLabView
            vocabulary={vocabulary}
            numStates={recognizer?.nStatesPerWord || 4}
          />
        )}
      </main>

      {/* Minimal Black & White Footer */}
      <footer className="border-t border-neutral-900 bg-black py-4 px-4 sm:px-6 mt-auto">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-neutral-500 font-mono">
          <span>HMM Speech Recognition Workbench · 13 MFCCs · Bakis Topology</span>
          <span>Forward · Viterbi Trellis · Baum-Welch (EM)</span>
        </div>
      </footer>
    </div>
  );
}
