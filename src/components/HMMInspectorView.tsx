import React, { useState, useMemo } from 'react';
import { ArrowRight, Layers, Activity, GitCommit, SlidersHorizontal } from 'lucide-react';
import { GaussianHMM, ViterbiResult } from '../lib/hmmEngine';
import { AudioSampleItem } from '../lib/presetDataset';

interface HMMInspectorViewProps {
  models: Map<string, GaussianHMM>;
  vocabulary: string[];
  currentSample: AudioSampleItem | null;
  selectedWord: string;
  onSelectWord: (word: string) => void;
}

export const HMMInspectorView: React.FC<HMMInspectorViewProps> = ({
  models,
  vocabulary,
  currentSample,
  selectedWord,
  onSelectWord,
}) => {
  const [hoveredFrame, setHoveredFrame] = useState<number | null>(null);
  const [hoveredState, setHoveredState] = useState<number | null>(null);

  const activeModel = models.get(selectedWord) || models.get(vocabulary[0]);

  // Compute Viterbi path and trellis for current sample against selected model
  const viterbiData: ViterbiResult | null = useMemo(() => {
    if (!activeModel || !currentSample?.pipelineResult?.mfccFeatures) return null;
    return activeModel.viterbi(currentSample.pipelineResult.mfccFeatures);
  }, [activeModel, currentSample]);

  const numStates = activeModel?.nStates || 4;
  const numFrames = currentSample?.pipelineResult?.mfccFeatures.length || 0;
  const timeAxis = currentSample?.pipelineResult?.timeAxisMs || [];

  return (
    <div className="space-y-6">
      {/* Top Controller Bar */}
      <div className="bg-[#0c0c0c] border border-neutral-800 rounded-xl p-4 sm:p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-neutral-900 border border-neutral-700 flex items-center justify-center text-white shrink-0">
            <Layers className="w-4 h-4" />
          </div>
          <div>
            <div className="text-[11px] font-mono text-neutral-500 uppercase tracking-wider">HMM Inspection & Trellis</div>
            <div className="text-sm font-semibold text-white flex items-center gap-2 mt-0.5">
              <span>Model:</span>
              <span className="font-mono text-white capitalize">{activeModel?.word || selectedWord}</span>
            </div>
          </div>
        </div>

        {/* Word Switcher Chips */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {vocabulary.map((w) => (
            <button
              key={w}
              onClick={() => onSelectWord(w)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono capitalize transition-colors border ${
                selectedWord.toLowerCase() === w.toLowerCase()
                  ? 'bg-white text-black border-white font-semibold'
                  : 'bg-neutral-950 border-neutral-800 text-neutral-400 hover:text-white hover:border-neutral-700'
              }`}
            >
              {w}
            </button>
          ))}
        </div>
      </div>

      {/* Main Feature: Viterbi Trellis Alignment Diagram in Pure Monochrome */}
      <div className="bg-[#0c0c0c] border border-neutral-800 rounded-xl p-5 sm:p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-neutral-800/80">
          <div>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Activity className="w-4 h-4 text-white" />
              <span>Viterbi Trellis Alignment Matrix</span>
            </h3>
            <p className="text-xs text-neutral-400 mt-0.5 font-mono">
              Optimal hidden state sequence q₁* ... q_T* mapped across acoustic MFCC frames
            </p>
          </div>

          {viterbiData && (
            <div className="flex items-center gap-3 text-xs font-mono text-neutral-400">
              <span>Viterbi Score: <strong className="text-white tabular-nums">{viterbiData.bestScore.toFixed(1)}</strong></span>
              <span>·</span>
              <span>Frames: <strong className="text-white tabular-nums">{numFrames}</strong></span>
            </div>
          )}
        </div>

        {/* Trellis Canvas Grid Container */}
        {viterbiData && numFrames > 0 ? (
          <div className="border border-neutral-800 rounded-lg bg-black p-4 overflow-x-auto">
            <div className="min-w-[640px]">
              <div className="space-y-4">
                {Array.from({ length: numStates }).map((_, stateIdx) => {
                  const stateLabel =
                    stateIdx === 0
                      ? 'S₀ (Onset)'
                      : stateIdx === numStates - 1
                      ? `S${stateIdx} (Coda)`
                      : `S${stateIdx} (Core)`;

                  return (
                    <div key={stateIdx} className="flex items-center gap-3">
                      <div className="w-24 shrink-0 text-right font-mono text-xs text-neutral-300 font-medium">
                        {stateLabel}
                      </div>

                      <div className="flex-1 flex items-center gap-1.5 relative py-1">
                        {/* Connecting line behind nodes */}
                        <div className="absolute inset-x-0 h-px bg-neutral-800 top-1/2 -translate-y-1/2" />

                        {Array.from({ length: Math.min(numFrames, 48) }).map((_, frameIdx) => {
                          const actualFrame = Math.floor((frameIdx / Math.min(numFrames, 48)) * numFrames);
                          const isViterbiPath = viterbiData.bestPath[actualFrame] === stateIdx;
                          const logProb = viterbiData.trellis[actualFrame]?.[stateIdx] ?? -Infinity;
                          const isHovered = hoveredFrame === actualFrame;

                          return (
                            <button
                              key={frameIdx}
                              onMouseEnter={() => {
                                setHoveredFrame(actualFrame);
                                setHoveredState(stateIdx);
                              }}
                              onMouseLeave={() => {
                                setHoveredFrame(null);
                                setHoveredState(null);
                              }}
                              className={`relative z-10 w-4 h-4 rounded-full transition-colors flex items-center justify-center shrink-0 ${
                                isViterbiPath
                                  ? 'bg-white ring-2 ring-neutral-400'
                                  : isFinite(logProb)
                                  ? 'bg-neutral-800 hover:bg-neutral-700'
                                  : 'bg-neutral-950 border border-neutral-800 opacity-30'
                              } ${isHovered ? 'ring-2 ring-white scale-125' : ''}`}
                              title={`t=${actualFrame} (${(timeAxis[actualFrame] || 0).toFixed(0)}ms), S=${stateIdx}, LogProb=${isFinite(logProb) ? logProb.toFixed(1) : '-inf'}`}
                            >
                              {isViterbiPath && <span className="w-1.5 h-1.5 rounded-full bg-black" />}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}

                {/* Time Axis Markers */}
                <div className="flex items-center gap-3 pt-2 border-t border-neutral-800 text-[10px] font-mono text-neutral-500">
                  <div className="w-24 shrink-0 text-right">Time (ms) →</div>
                  <div className="flex-1 flex justify-between px-1">
                    <span>0 ms</span>
                    <span>{((timeAxis[numFrames - 1] || 600) * 0.25).toFixed(0)} ms</span>
                    <span>{((timeAxis[numFrames - 1] || 600) * 0.5).toFixed(0)} ms</span>
                    <span>{((timeAxis[numFrames - 1] || 600) * 0.75).toFixed(0)} ms</span>
                    <span>{((timeAxis[numFrames - 1] || 600)).toFixed(0)} ms</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Hover Tooltip Strip */}
            <div className="mt-3.5 pt-3 border-t border-neutral-800 flex flex-wrap items-center justify-between text-xs font-mono text-neutral-300">
              <div>
                {hoveredFrame !== null && hoveredState !== null ? (
                  <span>
                    Frame <strong className="text-white">#{hoveredFrame}</strong> (≈ {(timeAxis[hoveredFrame] || 0).toFixed(0)}ms) · State <strong className="text-white">S{hoveredState}</strong> · Score: <strong className="text-white">{viterbiData.trellis[hoveredFrame]?.[hoveredState]?.toFixed(1) || '—'}</strong>
                  </span>
                ) : (
                  <span className="text-neutral-500">Hover over any trellis node to view exact frame log probability</span>
                )}
              </div>

              <div className="flex items-center gap-2 text-[11px] text-neutral-400">
                <span className="w-2 h-2 rounded-full bg-white inline-block" />
                <span>Active Viterbi Trajectory</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="p-8 text-center text-neutral-500 border border-dashed border-neutral-800 rounded-lg font-mono text-xs">
            Select an audio take above to visualize Viterbi trellis decoding.
          </div>
        )}
      </div>

      {/* Two Column Layout: Topology & State Profiles */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Left-to-Right Topology (5 cols) */}
        <div className="lg:col-span-5 bg-[#0c0c0c] border border-neutral-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-neutral-800/80">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <GitCommit className="w-4 h-4 text-white" />
              <span>Left-to-Right Bakis Topology</span>
            </h3>
            <span className="text-xs font-mono text-neutral-500">Matrix A</span>
          </div>

          <p className="text-xs text-neutral-400 font-mono leading-relaxed">
            Speech flows strictly forward in time. Diagonal self-loops (a_ii) model phoneme duration, while forward transitions (a_i,i+1) advance along acoustic stages.
          </p>

          {/* Monochrome State Flow */}
          <div className="py-3 flex items-center justify-between gap-1 overflow-x-auto">
            {Array.from({ length: numStates }).map((_, i) => {
              const selfLoop = activeModel?.transMat[i]?.[i] ?? 0.65;
              const nextTrans = activeModel?.transMat[i]?.[i + 1] ?? 0.35;

              return (
                <React.Fragment key={i}>
                  <div className="flex flex-col items-center gap-1.5 shrink-0">
                    <span className="text-[10px] font-mono text-neutral-300 bg-neutral-900 border border-neutral-800 px-2 py-0.5 rounded">
                      {(selfLoop * 100).toFixed(0)}%
                    </span>
                    <div className="w-9 h-9 rounded-lg bg-neutral-900 border border-neutral-700 flex items-center justify-center font-mono font-bold text-xs text-white">
                      S{i}
                    </div>
                    <span className="text-[10px] text-neutral-500 font-mono">
                      {i === 0 ? 'Onset' : i === numStates - 1 ? 'Coda' : 'Vowel'}
                    </span>
                  </div>

                  {i < numStates - 1 && (
                    <div className="flex-1 flex flex-col items-center px-1">
                      <span className="text-[10px] font-mono text-neutral-500">
                        {(nextTrans * 100).toFixed(0)}%
                      </span>
                      <ArrowRight className="w-3.5 h-3.5 text-neutral-600" />
                    </div>
                  )}
                </React.Fragment>
              );
            })}
          </div>

          {/* Clean Transition Table */}
          <div className="pt-2 border-t border-neutral-800/80 overflow-x-auto">
            <table className="w-full text-xs font-mono text-left">
              <thead>
                <tr className="border-b border-neutral-800 text-neutral-500">
                  <th className="py-1 px-2">From \ To</th>
                  {Array.from({ length: numStates }).map((_, j) => (
                    <th key={j} className="py-1 px-2">S{j}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-800/50">
                {Array.from({ length: numStates }).map((_, i) => (
                  <tr key={i} className="hover:bg-neutral-900/40">
                    <td className="py-1.5 px-2 text-neutral-400 font-semibold">S{i}</td>
                    {Array.from({ length: numStates }).map((_, j) => {
                      const val = activeModel?.transMat[i]?.[j] ?? 0;
                      return (
                        <td
                          key={j}
                          className={`py-1.5 px-2 tabular-nums ${
                            val > 0 ? 'text-white font-medium' : 'text-neutral-600'
                          }`}
                        >
                          {val.toFixed(2)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right: State Acoustic Emission Profiles (7 cols) */}
        <div className="lg:col-span-7 bg-[#0c0c0c] border border-neutral-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-neutral-800/80">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <SlidersHorizontal className="w-4 h-4 text-white" />
              <span>Gaussian Emission Profiles N(μ, σ²)</span>
            </h3>
            <span className="text-xs font-mono text-neutral-500">13 MFCC Coefficients</span>
          </div>

          <div className="space-y-2.5">
            {Array.from({ length: numStates }).map((_, stateIdx) => {
              const means = activeModel?.means[stateIdx] || [];

              return (
                <div key={stateIdx} className="p-3 rounded-lg bg-neutral-950 border border-neutral-800 space-y-2">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="font-semibold text-white">
                      State S{stateIdx} · {stateIdx === 0 ? 'Initial Onset' : stateIdx === numStates - 1 ? 'Ending Coda' : `Phoneme Core ${stateIdx}`}
                    </span>
                    <span className="text-neutral-500 text-[11px]">13 dimensions</span>
                  </div>

                  {/* Clean Monochrome 13 MFCC bars */}
                  <div className="flex items-end gap-1.5 h-12 px-2 bg-black rounded border border-neutral-800">
                    {means.slice(0, 13).map((meanVal, coeffIdx) => {
                      const clamped = Math.max(-15, Math.min(15, meanVal));
                      const heightPct = (Math.abs(clamped) / 15) * 85;
                      const isPositive = clamped >= 0;

                      return (
                        <div
                          key={coeffIdx}
                          className="flex-1 flex flex-col items-center justify-end h-full group relative"
                        >
                          <div
                            className={`w-full rounded-t-xs transition-colors ${
                              isPositive ? 'bg-white hover:bg-neutral-300' : 'bg-neutral-600 hover:bg-neutral-500'
                            }`}
                            style={{ height: `${Math.max(4, heightPct)}%` }}
                          />
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
