import React, { useRef, useEffect, useState } from 'react';
import { Play, Plus, CheckCircle2, ChevronRight, Activity } from 'lucide-react';
import { RecognitionResult } from '../lib/hmmEngine';
import { AudioSampleItem } from '../lib/presetDataset';

interface ClassifierViewProps {
  result: RecognitionResult | null;
  currentSample: AudioSampleItem | null;
  vocabulary: string[];
  samples: AudioSampleItem[];
  onAddWord: (word: string) => void;
  onRetrainAll: () => void;
  isRetraining: boolean;
  onPlayAudio: () => void;
  isPlaying: boolean;
  onSelectWordToInspect: (word: string) => void;
}

export const ClassifierView: React.FC<ClassifierViewProps> = ({
  result,
  currentSample,
  vocabulary,
  samples,
  onAddWord,
  onRetrainAll,
  isRetraining,
  onPlayAudio,
  isPlaying,
  onSelectWordToInspect,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [newWordInput, setNewWordInput] = useState('');

  // Minimal monochrome oscilloscope waveform renderer (Crisp white on pure black)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !currentSample?.rawSignal) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const signal = currentSample.rawSignal;

    // Solid matte black background
    ctx.fillStyle = '#050505';
    ctx.fillRect(0, 0, width, height);

    // Subtle center line
    ctx.strokeStyle = '#262626';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Pure crisp white waveform trace
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.3;
    ctx.beginPath();

    const step = Math.max(1, Math.floor(signal.length / width));
    for (let x = 0; x < width; x++) {
      const idx = Math.min(signal.length - 1, x * step);
      const val = signal[idx];
      const y = height / 2 - val * (height * 0.42);
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }, [currentSample]);

  const handleAddWordSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (newWordInput.trim()) {
      onAddWord(newWordInput.trim());
      setNewWordInput('');
    }
  };

  const isMatch =
    currentSample &&
    result &&
    currentSample.word.toLowerCase() === result.bestWord.toLowerCase();

  return (
    <div className="space-y-6">
      {/* Primary Recognition Result Header */}
      <div className="bg-[#0c0c0c] border border-neutral-800 rounded-xl p-5 sm:p-6 transition-colors">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 pb-5 border-b border-neutral-800/80">
          <div className="space-y-1.5">
            <div className="text-[11px] font-mono text-neutral-500 uppercase tracking-wider">
              Acoustic Classification · GMM-HMM
            </div>

            <div className="flex items-baseline gap-3">
              <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-white capitalize font-mono">
                {result?.bestWord || '—'}
              </h1>
              {result && (
                <span className="inline-flex items-center px-2.5 py-0.5 rounded text-xs font-mono bg-neutral-900 text-neutral-200 border border-neutral-700">
                  {(result.confidence * 100).toFixed(1)}% match
                </span>
              )}
            </div>

            {currentSample && (
              <div className="flex items-center gap-2 text-xs text-neutral-400 pt-1 font-mono">
                <span>Ground Truth: <strong className="text-white capitalize">{currentSample.word}</strong></span>
                <span>·</span>
                {isMatch ? (
                  <span className="text-white font-medium flex items-center gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5 text-white" />
                    <span>Exact Match</span>
                  </span>
                ) : (
                  <span className="text-neutral-400">
                    Acoustic divergence
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Metrics Column in pure monochrome */}
          {result && (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <div className="bg-neutral-950 border border-neutral-800 rounded-lg p-3 min-w-[110px]">
                <div className="text-[11px] font-mono text-neutral-500">Log-Likelihood</div>
                <div className="text-base font-bold font-mono text-white tabular-nums mt-0.5">
                  {result.scores[0]?.logLikelihood.toFixed(1) || '0'}
                </div>
                <div className="text-[10px] text-neutral-500 font-mono">log P(O|λ)</div>
              </div>

              <div className="bg-neutral-950 border border-neutral-800 rounded-lg p-3 min-w-[110px]">
                <div className="text-[11px] font-mono text-neutral-500">Frames (T)</div>
                <div className="text-base font-bold font-mono text-white tabular-nums mt-0.5">
                  {result.inputFramesCount}
                </div>
                <div className="text-[10px] text-neutral-500 font-mono">25ms hop 10ms</div>
              </div>

              <div className="bg-neutral-950 border border-neutral-800 rounded-lg p-3 min-w-[110px] col-span-2 sm:col-span-1">
                <div className="text-[11px] font-mono text-neutral-500">Models</div>
                <div className="text-base font-bold font-mono text-white tabular-nums mt-0.5">
                  {result.scores.length}
                </div>
                <div className="text-[10px] text-neutral-500 font-mono">Vocabulary</div>
              </div>
            </div>
          )}
        </div>

        {/* Minimal Monochrome Oscilloscope Waveform */}
        <div className="pt-4">
          <div className="flex items-center justify-between text-xs text-neutral-400 mb-2 font-mono">
            <span>Acoustic Waveform ({((currentSample?.durationSec || 0) * 1000).toFixed(0)} ms)</span>
            <button
              onClick={onPlayAudio}
              className="text-neutral-300 hover:text-white flex items-center gap-1 transition-colors font-medium text-[11px]"
            >
              <Play className={`w-3 h-3 ${isPlaying ? 'animate-spin' : ''}`} />
              <span>{isPlaying ? 'Auditioning...' : 'Play Audio'}</span>
            </button>
          </div>
          <div className="h-20 w-full rounded-lg border border-neutral-800 overflow-hidden bg-black relative">
            <canvas ref={canvasRef} className="w-full h-full block" />
          </div>
        </div>
      </div>

      {/* Two Column Layout: Model Ranking & Vocabulary Management */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Ranked Candidates (7 cols) */}
        <div className="lg:col-span-7 bg-[#0c0c0c] border border-neutral-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-neutral-800/80">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-white" />
              <h2 className="text-sm font-semibold text-white">Log-Likelihood Ranking</h2>
            </div>
            <span className="text-[11px] font-mono text-neutral-500">argmax_w log P(O|λ_w)</span>
          </div>

          <div className="space-y-2.5">
            {result?.scores.map((score, index) => {
              const isWinner = index === 0;
              const delta = score.logLikelihood - (result.scores[0]?.logLikelihood || 0);

              return (
                <div
                  key={score.word}
                  className={`p-3.5 rounded-lg border transition-colors ${
                    isWinner
                      ? 'bg-neutral-900 border-neutral-600'
                      : 'bg-neutral-950 border-neutral-800/80 hover:border-neutral-700'
                  }`}
                >
                  <div className="flex items-center justify-between text-xs mb-2">
                    <div className="flex items-center gap-2.5">
                      <span className={`font-mono font-bold capitalize text-sm ${
                        isWinner ? 'text-white' : 'text-neutral-300'
                      }`}>
                        {score.word}
                      </span>
                      {isWinner && (
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white text-black font-semibold">
                          PREDICTED
                        </span>
                      )}
                    </div>

                    <div className="flex items-center gap-3 font-mono text-right text-xs">
                      <span className="text-neutral-400 text-[11px]">
                        Δ: <span className={delta === 0 ? 'text-white font-semibold' : 'text-neutral-500'}>{delta.toFixed(1)}</span>
                      </span>
                      <span className="font-semibold text-white tabular-nums">
                        {score.logLikelihood.toFixed(1)}
                      </span>
                    </div>
                  </div>

                  {/* Clean Monochrome Progress Bar */}
                  <div className="w-full bg-neutral-950 h-1.5 rounded overflow-hidden border border-neutral-800">
                    <div
                      className={`h-full ${isWinner ? 'bg-white' : 'bg-neutral-600'}`}
                      style={{ width: `${Math.max(3, score.posterior * 100)}%` }}
                    />
                  </div>

                  <div className="flex items-center justify-between mt-2 pt-1 text-[11px] text-neutral-400 font-mono">
                    <span>
                      Confidence: <strong className="text-white tabular-nums">{(score.posterior * 100).toFixed(1)}%</strong>
                    </span>
                    <button
                      onClick={() => onSelectWordToInspect(score.word)}
                      className="text-neutral-300 hover:text-white flex items-center gap-0.5 transition-colors font-medium"
                    >
                      <span>Trellis</span>
                      <ChevronRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Column: Vocabulary & Dictionary Management (5 cols) */}
        <div className="lg:col-span-5 space-y-5">
          <div className="bg-[#0c0c0c] border border-neutral-800 rounded-xl p-5 space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-neutral-800/80">
              <div>
                <h2 className="text-sm font-semibold text-white">Active Vocabulary</h2>
                <p className="text-xs text-neutral-500 mt-0.5 font-mono">{vocabulary.length} isolated word models</p>
              </div>

              <button
                onClick={onRetrainAll}
                disabled={isRetraining}
                className="px-3 py-1.5 rounded-lg text-xs font-medium text-white bg-neutral-900 hover:bg-neutral-800 border border-neutral-700 transition-colors disabled:opacity-40"
              >
                {isRetraining ? 'Fitting...' : 'Re-train All'}
              </button>
            </div>

            <div className="space-y-2 font-mono">
              {vocabulary.map((word) => {
                const count = samples.filter((s) => s.word.toLowerCase() === word.toLowerCase()).length;
                return (
                  <div
                    key={word}
                    className="flex items-center justify-between p-2.5 rounded-lg bg-neutral-950 border border-neutral-800 text-xs"
                  >
                    <div className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-white" />
                      <span className="capitalize font-medium text-white">{word}</span>
                    </div>

                    <div className="flex items-center gap-3 text-neutral-400 text-[11px]">
                      <span>{count} take{count !== 1 ? 's' : ''}</span>
                      <button
                        onClick={() => onSelectWordToInspect(word)}
                        className="text-neutral-300 hover:text-white"
                      >
                        Inspect
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Add Custom Word Input */}
            <form onSubmit={handleAddWordSubmit} className="pt-2 border-t border-neutral-800 flex items-center gap-2">
              <input
                type="text"
                placeholder="Add word (e.g. echo)..."
                value={newWordInput}
                onChange={(e) => setNewWordInput(e.target.value)}
                className="bg-neutral-950 border border-neutral-800 rounded-lg px-3 py-1.5 text-xs text-white placeholder:text-neutral-600 font-mono focus:outline-none focus:border-neutral-600 transition-colors flex-1"
              />
              <button
                type="submit"
                className="px-3.5 py-1.5 bg-white text-black hover:bg-neutral-200 rounded-lg text-xs font-medium transition-colors flex items-center gap-1"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Add</span>
              </button>
            </form>
          </div>

          {/* Minimal Explainer Card */}
          <div className="bg-[#0c0c0c] border border-neutral-800 rounded-xl p-5 space-y-2 text-xs text-neutral-400 font-mono leading-relaxed">
            <div className="text-white font-semibold">Left-to-Right Acoustic Flow</div>
            <p>
              States model sequential phonetic phases. The Forward algorithm calculates exact log-likelihood log P(O|λ_w), and the Viterbi algorithm backtracks the optimal state progression sequence.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
