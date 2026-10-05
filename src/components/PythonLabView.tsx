import React, { useState } from 'react';
import { Copy, Check, Download, Play, Code2 } from 'lucide-react';
import { generateHmmlearnPythonCode, generatePureNumpyPythonCode } from '../lib/pythonCodeGenerator';

interface PythonLabViewProps {
  vocabulary: string[];
  numStates: number;
}

export const PythonLabView: React.FC<PythonLabViewProps> = ({ vocabulary, numStates }) => {
  const [activeTab, setActiveTab] = useState<'hmmlearn' | 'numpy'>('hmmlearn');
  const [copied, setCopied] = useState(false);
  const [terminalOutput, setTerminalOutput] = useState<string[] | null>(null);

  const hmmlearnCode = generateHmmlearnPythonCode(vocabulary, numStates);
  const numpyCode = generatePureNumpyPythonCode(vocabulary, numStates);
  const activeCode = activeTab === 'hmmlearn' ? hmmlearnCode : numpyCode;

  const handleCopy = () => {
    navigator.clipboard.writeText(activeCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const filename = activeTab === 'hmmlearn' ? 'speech_hmmlearn.py' : 'speech_numpy_hmm.py';
    const blob = new Blob([activeCode], { type: 'text/x-python;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleSimulateExecution = () => {
    setTerminalOutput([
      `$ python ${activeTab === 'hmmlearn' ? 'speech_hmmlearn.py' : 'speech_numpy_hmm.py'}`,
      'Extracting 13 MFCC coefficients from audio samples...',
      ...vocabulary.map((w) => `  ✓ Processed takes for '${w}' (16kHz PCM mono)`),
      'Fitting Gaussian HMM models with Baum-Welch (EM)...',
      ...vocabulary.map((w) => `  ✓ Model '${w}' converged`),
      `Scoring test utterance against models:`,
      ...vocabulary.map((w, idx) => `  Word: ${w.padEnd(8)} | Log-Likelihood: ${(-1340 - idx * 170).toFixed(1)}`),
      `Prediction: '${vocabulary[0].toUpperCase()}' (Viterbi sequence: [0, 0, 1, 1, 2, 2, 3, 3]...)`,
      'Status: Success (Execution time: 0.38s)',
    ]);
  };

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-[#0c0c0c] border border-neutral-800 rounded-xl p-5 sm:p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-neutral-800/80">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-neutral-900 border border-neutral-700 flex items-center justify-center text-white shrink-0">
              <Code2 className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-white">Production Python HMM Implementation</h2>
              <p className="text-xs text-neutral-400 mt-0.5 font-mono">
                Synchronized with {vocabulary.length} vocabulary words and {numStates} hidden states
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 self-start sm:self-auto">
            <button
              onClick={handleSimulateExecution}
              className="px-3 py-1.5 bg-neutral-900 hover:bg-neutral-800 text-white rounded-lg text-xs font-medium border border-neutral-700 transition-colors flex items-center gap-1.5"
            >
              <Play className="w-3.5 h-3.5" />
              <span>Simulate Run</span>
            </button>

            <button
              onClick={handleCopy}
              className="px-3 py-1.5 bg-neutral-900 hover:bg-neutral-800 text-white rounded-lg text-xs font-medium border border-neutral-700 transition-colors flex items-center gap-1.5"
            >
              {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? 'Copied' : 'Copy'}</span>
            </button>

            <button
              onClick={handleDownload}
              className="px-3.5 py-1.5 bg-white hover:bg-neutral-200 text-black font-semibold rounded-lg text-xs transition-colors flex items-center gap-1.5"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download .py</span>
            </button>
          </div>
        </div>

        {/* Minimal Code Tabs */}
        <div className="flex items-center gap-2 pt-4">
          <button
            onClick={() => setActiveTab('hmmlearn')}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-colors border ${
              activeTab === 'hmmlearn'
                ? 'bg-white text-black border-white font-semibold'
                : 'bg-neutral-950 border-neutral-800 text-neutral-400 hover:text-white hover:border-neutral-700'
            }`}
          >
            hmmlearn + librosa (Standard)
          </button>
          <button
            onClick={() => setActiveTab('numpy')}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-colors border ${
              activeTab === 'numpy'
                ? 'bg-white text-black border-white font-semibold'
                : 'bg-neutral-950 border-neutral-800 text-neutral-400 hover:text-white hover:border-neutral-700'
            }`}
          >
            Pure NumPy / SciPy (Zero Dependencies)
          </button>
        </div>
      </div>

      {/* Terminal Output */}
      {terminalOutput && (
        <div className="bg-[#050505] border border-neutral-800 rounded-xl p-4 font-mono text-xs">
          <div className="flex items-center justify-between text-neutral-500 pb-2 border-b border-neutral-900 text-[11px]">
            <span className="text-white">Terminal Output</span>
            <button onClick={() => setTerminalOutput(null)} className="text-neutral-500 hover:text-neutral-300">
              Clear
            </button>
          </div>
          <div className="pt-2.5 space-y-1 text-neutral-300">
            {terminalOutput.map((l, i) => (
              <div key={i} className={l.startsWith('$') ? 'text-white font-bold' : ''}>
                {l}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Code Viewer Container */}
      <div className="bg-[#070707] border border-neutral-800 rounded-xl overflow-hidden font-mono text-xs">
        <div className="bg-neutral-950 px-4 py-2.5 border-b border-neutral-800 flex items-center justify-between text-[11px] text-neutral-400">
          <span>{activeTab === 'hmmlearn' ? 'speech_recognizer_hmmlearn.py' : 'pure_numpy_speech_hmm.py'}</span>
          <span>Python 3.8+ · UTF-8</span>
        </div>
        <div className="p-4 overflow-x-auto max-h-[550px] leading-relaxed text-neutral-300 selection:bg-neutral-800">
          <pre>
            <code>{activeCode}</code>
          </pre>
        </div>
      </div>
    </div>
  );
};
