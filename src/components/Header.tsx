import React from 'react';
import { Mic, Activity, Layers, Sliders, Cpu, Code2 } from 'lucide-react';

interface HeaderProps {
  activeTab: 'classifier' | 'trellis' | 'mfcc' | 'training' | 'python';
  onTabChange: (tab: 'classifier' | 'trellis' | 'mfcc' | 'training' | 'python') => void;
  isRecording: boolean;
  onToggleRecord: () => void;
  onQuickSimulate: () => void;
  isSimulating: boolean;
}

const TABS = [
  { id: 'classifier' as const, label: 'Classifier', icon: Activity },
  { id: 'trellis' as const, label: 'Trellis', icon: Layers },
  { id: 'mfcc' as const, label: 'MFCC', icon: Sliders },
  { id: 'training' as const, label: 'Baum-Welch', icon: Cpu },
  { id: 'python' as const, label: 'Python', icon: Code2 },
];

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onTabChange,
  isRecording,
  onToggleRecord,
  onQuickSimulate,
  isSimulating,
}) => {
  return (
    <header className="sticky top-0 z-40 bg-black/90 backdrop-blur-md border-b border-neutral-800">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center gap-2.5 shrink-0">
          <div className="w-7 h-7 rounded-lg bg-neutral-900 border border-neutral-700 flex items-center justify-center text-white font-mono text-xs font-bold">
            H
          </div>
          <span className="font-semibold tracking-tight text-sm text-white">
            HMM Speech Lab
          </span>
        </div>

        {/* Minimal Black & White Navigation */}
        <nav className="flex items-center bg-neutral-950 p-1 rounded-lg border border-neutral-800 overflow-x-auto max-w-full">
          {TABS.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-colors flex items-center gap-1.5 whitespace-nowrap ${
                  isActive
                    ? 'bg-white text-black shadow-sm font-semibold'
                    : 'text-neutral-400 hover:text-white hover:bg-neutral-900'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Right Actions */}
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={onQuickSimulate}
            disabled={isSimulating}
            className="hidden sm:inline-flex items-center px-3 py-1.5 rounded-lg text-xs font-medium text-neutral-300 hover:text-white bg-neutral-900 hover:bg-neutral-800 border border-neutral-800 transition-colors whitespace-nowrap disabled:opacity-40"
          >
            Random Take
          </button>

          <button
            onClick={onToggleRecord}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5 whitespace-nowrap ${
              isRecording
                ? 'bg-neutral-200 text-black border border-white animate-pulse'
                : 'bg-white text-black hover:bg-neutral-200'
            }`}
          >
            <Mic className="w-3.5 h-3.5" />
            <span>{isRecording ? 'Listening...' : 'Record Voice'}</span>
          </button>
        </div>
      </div>
    </header>
  );
};
