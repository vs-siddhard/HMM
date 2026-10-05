import React, { useState } from 'react';
import { Play, Cpu, CheckCircle2, TrendingUp } from 'lucide-react';
import { SpeechRecognizer, HMMTrainingEpoch } from '../lib/hmmEngine';
import { AudioSampleItem } from '../lib/presetDataset';

interface TrainingLabViewProps {
  recognizer: SpeechRecognizer;
  samples: AudioSampleItem[];
  vocabulary: string[];
  onModelRetrained: () => void;
}

export const TrainingLabView: React.FC<TrainingLabViewProps> = ({
  recognizer,
  samples,
  vocabulary,
  onModelRetrained,
}) => {
  const [selectedWord, setSelectedWord] = useState<string>(vocabulary[0] || 'apple');
  const [numStates, setNumStates] = useState<number>(4);
  const [maxIterations, setMaxIterations] = useState<number>(10);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [history, setHistory] = useState<HMMTrainingEpoch[]>([]);
  const [message, setMessage] = useState<string | null>(null);

  const wordSamples = samples.filter((s) => s.word.toLowerCase() === selectedWord.toLowerCase());

  const handleStartTraining = () => {
    setIsTraining(true);
    setMessage(null);

    setTimeout(() => {
      const dataList: number[][][] = [];
      for (const sample of wordSamples) {
        if (sample.pipelineResult?.mfccFeatures) {
          dataList.push(sample.pipelineResult.mfccFeatures);
        }
      }

      if (dataList.length === 0) {
        setIsTraining(false);
        setMessage('No audio samples available for this word.');
        return;
      }

      const model = recognizer.addWord(selectedWord);
      model.nStates = numStates;

      const hist = model.fitBaumWelch(dataList, maxIterations, 1e-3);
      setHistory(hist);

      setIsTraining(false);
      setMessage(`Model for '${selectedWord.toUpperCase()}' converged across ${hist.length} epochs.`);
      onModelRetrained();
    }, 150);
  };

  return (
    <div className="space-y-6">
      {/* Top Banner Card */}
      <div className="bg-[#0c0c0c] border border-neutral-800 rounded-xl p-5 sm:p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-neutral-800/80">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-neutral-900 border border-neutral-700 flex items-center justify-center text-white shrink-0">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-white">Baum-Welch (EM) Training Workbench</h2>
              <p className="text-xs text-neutral-400 mt-0.5 font-mono">
                Optimizes state transition matrix A and emission parameters μ, σ² using forward-backward probabilities (α, β)
              </p>
            </div>
          </div>

          <button
            onClick={handleStartTraining}
            disabled={isTraining || wordSamples.length === 0}
            className="px-4 py-2 bg-white text-black hover:bg-neutral-200 font-semibold text-xs rounded-lg transition-colors flex items-center gap-2 disabled:opacity-40 self-start sm:self-auto"
          >
            <Play className={`w-3.5 h-3.5 ${isTraining ? 'animate-spin' : ''}`} />
            <span>{isTraining ? 'Optimizing...' : `Train '${selectedWord.toUpperCase()}' HMM`}</span>
          </button>
        </div>

        {/* Hyperparameters Controls in pure monochrome */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 pt-5 text-xs font-mono text-neutral-400">
          <div>
            <span className="text-white block text-xs font-semibold mb-2 font-sans">Target Word:</span>
            <div className="flex flex-wrap gap-1.5">
              {vocabulary.map((w) => (
                <button
                  key={w}
                  onClick={() => {
                    setSelectedWord(w);
                    setMessage(null);
                  }}
                  className={`px-3 py-1.5 rounded-lg capitalize transition-colors border ${
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

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-white font-sans font-semibold">Hidden States (N):</span>
              <span className="text-white font-bold">{numStates} States</span>
            </div>
            <input
              type="range"
              min="3"
              max="6"
              value={numStates}
              onChange={(e) => setNumStates(parseInt(e.target.value, 10))}
              className="w-full accent-white cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-neutral-500 mt-1">
              <span>3 (Short)</span>
              <span>4 (Default)</span>
              <span>5 (Multi-syllable)</span>
              <span>6 (Complex)</span>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-white font-sans font-semibold">Max Epochs:</span>
              <span className="text-white font-bold">{maxIterations} Iterations</span>
            </div>
            <input
              type="range"
              min="3"
              max="25"
              value={maxIterations}
              onChange={(e) => setMaxIterations(parseInt(e.target.value, 10))}
              className="w-full accent-white cursor-pointer"
            />
            <div className="text-[10px] text-neutral-500 mt-1">Convergence tolerance: 10⁻³</div>
          </div>
        </div>
      </div>

      {/* Convergence History Table in Pure Monochrome */}
      <div className="bg-[#0c0c0c] border border-neutral-800 rounded-xl p-5 sm:p-6 space-y-3">
        <div className="flex items-center justify-between pb-3 border-b border-neutral-800/80 font-mono text-xs">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-white" />
            <span className="font-semibold text-white font-sans">Convergence Trajectory</span>
          </div>
          {message && (
            <span className="text-white flex items-center gap-1.5 font-medium">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>{message}</span>
            </span>
          )}
        </div>

        {history.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full font-mono text-xs text-left">
              <thead>
                <tr className="border-b border-neutral-800 text-neutral-500 text-[11px]">
                  <th className="py-2 px-3">Epoch</th>
                  <th className="py-2 px-3">Log-Likelihood log P(O|λ)</th>
                  <th className="py-2 px-3">Delta |ΔL|</th>
                  <th className="py-2 px-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-800/50">
                {history.map((row) => (
                  <tr key={row.epoch} className="hover:bg-neutral-900/40">
                    <td className="py-2.5 px-3 text-white font-bold">#{row.epoch}</td>
                    <td className="py-2.5 px-3 text-neutral-200 tabular-nums">
                      {row.totalLogLikelihood.toFixed(2)}
                    </td>
                    <td className="py-2.5 px-3 text-neutral-400 tabular-nums">
                      {row.delta.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3">
                      <span className="text-[10px] px-2 py-0.5 rounded bg-neutral-900 text-neutral-200 border border-neutral-700 font-semibold">
                        Optimized
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="py-8 text-center text-neutral-500 text-xs font-mono">
            Select word parameters and click &quot;Train HMM&quot; above to run Baum-Welch iterations.
          </div>
        )}
      </div>
    </div>
  );
};
