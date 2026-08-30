import React from "react";
import { Cpu, Satellite, Layers, AlertCircle } from "lucide-react";

interface HeaderProps {
  apiOnline: boolean;
}

export const Header: React.FC<HeaderProps> = ({ apiOnline }) => {
  return (
    <header className="bg-slate-900 border-b border-slate-800 px-6 py-3.5 flex flex-wrap items-center justify-between gap-4 select-none">
      <div className="flex items-center gap-3">
        <div className="h-9 w-9 rounded-lg bg-emerald-950 border border-emerald-500/30 flex items-center justify-center text-emerald-400 shadow-sm">
          <Satellite className="h-5 w-5" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-100 text-base tracking-tight">
              UrbanCadastral <span className="text-emerald-400 font-mono">AI-ML</span>
            </span>
            <span className="text-[10px] font-mono font-medium uppercase px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
              Inference Workstation
            </span>
          </div>
          <p className="text-xs text-slate-400 tracking-wide">
            Real LightUNet Aerial Building Footprint Extraction & Polygonization
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2.5 flex-wrap text-xs">
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60 text-slate-300">
          <Layers className="h-3.5 w-3.5 text-indigo-400" />
          <span className="font-mono text-slate-400">Architecture:</span>
          <span className="font-medium text-slate-200">LightUNet (1.94M)</span>
        </div>

        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60 text-slate-300">
          <Cpu className="h-3.5 w-3.5 text-amber-400" />
          <span className="font-mono text-slate-400">Runtime:</span>
          <span className="font-medium text-slate-200">CPU (4 Threads)</span>
        </div>

        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60 text-slate-300">
          <span className="font-mono text-slate-400">Domain:</span>
          <span className="font-medium text-slate-200">SpaceNet 2 (30cm GSD)</span>
        </div>

        <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md border font-medium ${
          apiOnline
            ? "bg-emerald-950/40 border-emerald-700/50 text-emerald-300"
            : "bg-rose-950/40 border-rose-700/50 text-rose-300"
        }`}>
          {apiOnline ? (
            <>
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span>API Active</span>
            </>
          ) : (
            <>
              <AlertCircle className="h-3.5 w-3.5" />
              <span>API Offline</span>
            </>
          )}
        </div>
      </div>
    </header>
  );
};
