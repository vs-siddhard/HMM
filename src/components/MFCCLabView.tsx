import React, { useState, useRef, useEffect } from 'react';
import { Sliders } from 'lucide-react';
import { AudioSampleItem } from '../lib/presetDataset';

interface MFCCLabViewProps {
  currentSample: AudioSampleItem | null;
}

export const MFCCLabView: React.FC<MFCCLabViewProps> = ({ currentSample }) => {
  const [selectedFrameIdx, setSelectedFrameIdx] = useState<number>(10);
  const [activeStage, setActiveStage] = useState<'mfcc' | 'fft'>('mfcc');
  const mfccCanvasRef = useRef<HTMLCanvasElement>(null);
  const fftCanvasRef = useRef<HTMLCanvasElement>(null);

  const pipeline = currentSample?.pipelineResult;
  const numFrames = pipeline?.frames.length || 0;
  const timeAxis = pipeline?.timeAxisMs || [];

  const safeFrameIdx = Math.max(0, Math.min(numFrames - 1, selectedFrameIdx));

  // Pure Monochrome Grayscale MFCC Heatmap (No neon colors)
  useEffect(() => {
    const canvas = mfccCanvasRef.current;
    if (!canvas || !pipeline?.mfccFeatures || pipeline.mfccFeatures.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const features = pipeline.mfccFeatures;
    const T = features.length;
    const D = 13;

    // Pure matte black background
    ctx.fillStyle = '#050505';
    ctx.fillRect(0, 0, width, height);

    const cellWidth = width / T;
    const cellHeight = height / D;

    let minVal = -20;
    let maxVal = 20;
    for (let t = 0; t < T; t++) {
      for (let d = 1; d < D; d++) {
        const val = features[t][d];
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }
    }

    for (let t = 0; t < T; t++) {
      for (let d = 0; d < D; d++) {
        const val = features[t][d];
        const norm = Math.max(0, Math.min(1, (val - minVal) / (maxVal - minVal || 1)));

        // Pure monochrome grayscale tone: dark charcoal (15) to pure white (255)
        const tone = Math.floor(12 + norm * 243);
        ctx.fillStyle = `rgb(${tone}, ${tone}, ${tone})`;
        ctx.fillRect(t * cellWidth, d * cellHeight, Math.ceil(cellWidth), Math.ceil(cellHeight));
      }
    }

    // Clean white scrub cursor line
    if (safeFrameIdx >= 0 && safeFrameIdx < T) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(safeFrameIdx * cellWidth + cellWidth / 2, 0);
      ctx.lineTo(safeFrameIdx * cellWidth + cellWidth / 2, height);
      ctx.stroke();
    }
  }, [pipeline, safeFrameIdx]);

  // Pure Monochrome Grayscale FFT Spectrogram
  useEffect(() => {
    const canvas = fftCanvasRef.current;
    if (!canvas || !pipeline?.fftSpectrogram || pipeline.fftSpectrogram.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const spectrogram = pipeline.fftSpectrogram;
    const T = spectrogram.length;
    const bins = Math.min(128, spectrogram[0].length);

    ctx.fillStyle = '#050505';
    ctx.fillRect(0, 0, width, height);

    const cellWidth = width / T;
    const cellHeight = height / bins;

    for (let t = 0; t < T; t++) {
      for (let b = 0; b < bins; b++) {
        const yBin = bins - 1 - b;
        const mag = spectrogram[t][b];
        const logMag = Math.log10(1 + mag * 50);
        const norm = Math.max(0, Math.min(1, logMag / 2.2));

        const tone = Math.floor(norm * 255);
        ctx.fillStyle = `rgb(${tone}, ${tone}, ${tone})`;
        ctx.fillRect(t * cellWidth, yBin * cellHeight, Math.ceil(cellWidth), Math.ceil(cellHeight));
      }
    }
  }, [pipeline]);

  const currentFrameFeatures = pipeline?.mfccFeatures[safeFrameIdx] || [];

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-[#0c0c0c] border border-neutral-800 rounded-xl p-5 sm:p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-neutral-800/80">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-neutral-900 border border-neutral-700 flex items-center justify-center text-white shrink-0">
              <Sliders className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-white">MFCC Feature Extraction Pipeline</h2>
              <p className="text-xs text-neutral-400 mt-0.5 font-mono">
                Pre-emphasis (0.97) · Framing (25ms/10ms) · Hamming · FFT (512) · Mel (26) · DCT-II (13)
              </p>
            </div>
          </div>

          <div className="flex items-center bg-neutral-950 p-1 rounded-lg border border-neutral-800 self-start sm:self-auto">
            <button
              onClick={() => setActiveStage('mfcc')}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                activeStage === 'mfcc' ? 'bg-white text-black font-semibold' : 'text-neutral-400 hover:text-white'
              }`}
            >
              13-MFCC Matrix
            </button>
            <button
              onClick={() => setActiveStage('fft')}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                activeStage === 'fft' ? 'bg-white text-black font-semibold' : 'text-neutral-400 hover:text-white'
              }`}
            >
              FFT Spectrogram
            </button>
          </div>
        </div>

        {/* Heatmap Area */}
        <div className="pt-4 space-y-3">
          <div className="flex items-center justify-between text-xs font-mono text-neutral-400">
            <span>{activeStage === 'mfcc' ? '13 Coefficients (Y) vs Time Frames (X)' : 'Frequency Spectrum 0–4kHz (Y) vs Time (X)'}</span>
            <span>
              Frame <strong className="text-white">#{safeFrameIdx}</strong> ({(timeAxis[safeFrameIdx] || 0).toFixed(0)} ms)
            </span>
          </div>

          <div className="h-48 w-full rounded-lg border border-neutral-800 overflow-hidden bg-black">
            {activeStage === 'mfcc' ? (
              <canvas
                ref={mfccCanvasRef}
                onClick={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  const clickX = e.clientX - rect.left;
                  const frame = Math.floor((clickX / rect.width) * numFrames);
                  setSelectedFrameIdx(frame);
                }}
                className="w-full h-full block cursor-crosshair"
              />
            ) : (
              <canvas ref={fftCanvasRef} className="w-full h-full block" />
            )}
          </div>

          {/* Minimal Monochrome Scrub Slider */}
          <div className="flex items-center gap-3 pt-2">
            <span className="font-mono text-xs text-neutral-400 shrink-0">Frame Scrub:</span>
            <input
              type="range"
              min="0"
              max={Math.max(1, numFrames - 1)}
              value={safeFrameIdx}
              onChange={(e) => setSelectedFrameIdx(parseInt(e.target.value, 10))}
              className="flex-1 accent-white cursor-pointer"
            />
            <span className="font-mono text-xs text-white tabular-nums shrink-0">
              {safeFrameIdx} / {Math.max(0, numFrames - 1)}
            </span>
          </div>
        </div>
      </div>

      {/* Frame Vector Readout in Pure Monochrome */}
      <div className="bg-[#0c0c0c] border border-neutral-800 rounded-xl p-5 sm:p-6 space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-neutral-800/80 font-mono text-xs">
          <span className="font-semibold text-white">Frame #{safeFrameIdx} Feature Vector (13 Dimensions)</span>
          <span className="text-neutral-500">t = {(timeAxis[safeFrameIdx] || 0).toFixed(0)} ms</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-2.5 pt-1">
          {currentFrameFeatures.map((val, idx) => (
            <div
              key={idx}
              className="p-3 rounded-lg bg-neutral-950 border border-neutral-800 font-mono text-xs flex flex-col justify-between"
            >
              <div className="text-neutral-500 text-[11px]">{idx === 0 ? 'C₀ (Energy)' : `C${idx}`}</div>
              <div className="text-sm font-bold tabular-nums mt-1 text-white">
                {val.toFixed(2)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
