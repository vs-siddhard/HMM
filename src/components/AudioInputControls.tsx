import React, { useRef, useState } from 'react';
import { Upload, Play, Sparkles, Volume2 } from 'lucide-react';
import { AudioSampleItem } from '../lib/presetDataset';

interface AudioInputControlsProps {
  currentSample: AudioSampleItem | null;
  onSelectSample: (sample: AudioSampleItem) => void;
  samples: AudioSampleItem[];
  vocabulary: string[];
  isRecording: boolean;
  onStartRecord: () => void;
  onStopRecord: () => void;
  onSynthesizeCustom: (word: string, duration: number) => void;
  onUploadAudio: (file: File) => void;
  onPlayAudio: () => void;
  isPlaying: boolean;
}

export const AudioInputControls: React.FC<AudioInputControlsProps> = ({
  currentSample,
  onSelectSample,
  samples,
  vocabulary,
  onSynthesizeCustom,
  onUploadAudio,
  onPlayAudio,
  isPlaying,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [synthWord, setSynthWord] = useState('apple');
  const [synthDuration, setSynthDuration] = useState(0.65);
  const [activeTab, setActiveTab] = useState<'presets' | 'synth' | 'upload'>('presets');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onUploadAudio(file);
    }
  };

  return (
    <div className="bg-[#0c0c0c] border border-neutral-800 rounded-xl p-4 sm:p-5 transition-colors">
      {/* Top Header Bar inside card */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3.5 border-b border-neutral-800/80">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-neutral-900 border border-neutral-800 flex items-center justify-center text-neutral-300 shrink-0">
            <Volume2 className="w-4 h-4 text-white" />
          </div>

          <div>
            <div className="text-[11px] font-mono text-neutral-500 uppercase tracking-wider">
              Audio Input · {((currentSample?.durationSec || 0) * 1000).toFixed(0)}ms · 16kHz
            </div>
            <div className="text-sm font-semibold text-white flex items-center gap-2 mt-0.5">
              <span className="capitalize font-mono">{currentSample?.word || 'None'}</span>
              <span className="text-xs font-normal text-neutral-400 font-mono">
                {currentSample?.label.includes('Take') ? currentSample.label.split('·')[1].trim() : ''}
              </span>
            </div>
          </div>
        </div>

        {/* Mode Switcher & Play Button */}
        <div className="flex items-center gap-2 self-start sm:self-auto">
          {/* Segmented Mode Toggle */}
          <div className="flex items-center bg-neutral-950 p-1 rounded-lg border border-neutral-800">
            <button
              onClick={() => setActiveTab('presets')}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                activeTab === 'presets'
                  ? 'bg-white text-black shadow-sm font-semibold'
                  : 'text-neutral-400 hover:text-white'
              }`}
            >
              Presets
            </button>
            <button
              onClick={() => setActiveTab('synth')}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                activeTab === 'synth'
                  ? 'bg-white text-black shadow-sm font-semibold'
                  : 'text-neutral-400 hover:text-white'
              }`}
            >
              Synth
            </button>
            <button
              onClick={() => setActiveTab('upload')}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                activeTab === 'upload'
                  ? 'bg-white text-black shadow-sm font-semibold'
                  : 'text-neutral-400 hover:text-white'
              }`}
            >
              Upload
            </button>
          </div>

          {/* Minimal Play Button */}
          <button
            onClick={onPlayAudio}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium border transition-colors flex items-center gap-1.5 ${
              isPlaying
                ? 'bg-white text-black border-white'
                : 'bg-neutral-900 hover:bg-neutral-800 text-white border-neutral-700'
            }`}
          >
            <Play className={`w-3.5 h-3.5 ${isPlaying ? 'animate-spin' : ''}`} />
            <span>{isPlaying ? 'Playing...' : 'Play'}</span>
          </button>
        </div>
      </div>

      {/* Mode Sub-Panels */}
      <div className="pt-3.5">
        {activeTab === 'presets' && (
          <div className="flex flex-wrap gap-2 items-center">
            {samples.map((s) => {
              const isSelected = currentSample?.id === s.id;
              return (
                <button
                  key={s.id}
                  onClick={() => onSelectSample(s)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-colors flex items-center gap-2 border ${
                    isSelected
                      ? 'bg-white text-black border-white font-semibold'
                      : 'bg-neutral-950 border-neutral-800 text-neutral-400 hover:text-white hover:border-neutral-700'
                  }`}
                >
                  <span className="capitalize">{s.word}</span>
                  <span className={`text-[10px] px-1 py-0.2 rounded ${
                    isSelected ? 'bg-neutral-200 text-neutral-900' : 'bg-neutral-900 text-neutral-500'
                  }`}>
                    {s.label.includes('Take') ? s.label.split('·')[1].trim() : `${(s.durationSec * 1000).toFixed(0)}ms`}
                  </span>
                </button>
              );
            })}
          </div>
        )}

        {activeTab === 'synth' && (
          <div className="flex flex-wrap items-center gap-4 py-1">
            <div className="flex items-center gap-2">
              <span className="text-xs text-neutral-400 font-mono">Word:</span>
              <select
                value={synthWord}
                onChange={(e) => setSynthWord(e.target.value)}
                className="bg-neutral-900 border border-neutral-800 rounded-lg px-2.5 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-neutral-600"
              >
                {vocabulary.map((w) => (
                  <option key={w} value={w}>
                    {w}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-center gap-2.5">
              <span className="text-xs text-neutral-400 font-mono">Duration:</span>
              <input
                type="range"
                min="0.35"
                max="1.1"
                step="0.05"
                value={synthDuration}
                onChange={(e) => setSynthDuration(parseFloat(e.target.value))}
                className="w-28 accent-white"
              />
              <span className="text-xs font-mono text-neutral-300 tabular-nums">{(synthDuration * 1000).toFixed(0)}ms</span>
            </div>

            <button
              onClick={() => onSynthesizeCustom(synthWord, synthDuration)}
              className="px-3.5 py-1.5 bg-neutral-900 hover:bg-neutral-800 text-white rounded-lg text-xs font-medium border border-neutral-700 transition-colors flex items-center gap-1.5"
            >
              <Sparkles className="w-3.5 h-3.5 text-neutral-300" />
              <span>Generate Formants</span>
            </button>
          </div>
        )}

        {activeTab === 'upload' && (
          <div className="flex flex-wrap items-center gap-3 py-1">
            <input
              type="file"
              ref={fileInputRef}
              accept="audio/*,.wav,.mp3"
              onChange={handleFileChange}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-4 py-2 border border-neutral-700 hover:border-neutral-500 rounded-lg text-xs font-medium text-neutral-200 hover:text-white bg-neutral-900 hover:bg-neutral-800 transition-colors flex items-center gap-2"
            >
              <Upload className="w-3.5 h-3.5" />
              <span>Select Audio File (.wav, .mp3)</span>
            </button>
            <span className="text-xs text-neutral-500 font-mono">Automatically normalized to 16kHz PCM</span>
          </div>
        )}
      </div>
    </div>
  );
};
