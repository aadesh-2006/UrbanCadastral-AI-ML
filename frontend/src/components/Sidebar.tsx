import React from "react";
import {
  UploadCloud,
  Zap,
  CheckCircle2,
  Sparkles,
  Download,
  Info,
  Building2,
  RotateCcw
} from "lucide-react";
import type { InferenceResult, PresetItem, ProcessingStage, GeoJSONFeature } from "../types";

interface SidebarProps {
  presets: PresetItem[];
  selectedPresetId: string | null;
  onSelectPreset: (id: string) => void;
  onFileUpload: (file: File) => void;
  selectedFile: File | null;
  currentImageInfo: {
    name: string;
    dimensions?: string;
    format: string;
    georeferenced: boolean;
    crs?: string;
  } | null;
  onRunInference: () => void;
  processingStage: ProcessingStage;
  result: InferenceResult | null;
  selectedBuilding: GeoJSONFeature | null;
  onDownloadGeoJSON: () => void;
  onReset: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  presets,
  selectedPresetId,
  onSelectPreset,
  onFileUpload,
  selectedFile,
  currentImageInfo,
  onRunInference,
  processingStage,
  result,
  selectedBuilding,
  onDownloadGeoJSON,
  onReset
}) => {
  const isProcessing = processingStage !== "idle" && processingStage !== "complete" && processingStage !== "error";

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileUpload(e.target.files[0]);
    }
  };

  const stages = [
    { key: "preprocessing", label: "Image Preprocessing (Normalized to [0,1])" },
    { key: "segmentation", label: "LightUNet Deep Learning Forward Pass" },
    { key: "cleanup", label: "Morphological Boundary Cleanup & Filtering" },
    { key: "polygonization", label: "Contour Simplification (Ramer-Douglas-Peucker)" },
    { key: "geojson", label: "Geospatial Affine Coordinate Transformation" }
  ];

  return (
    <aside className="w-80 md:w-96 flex-shrink-0 bg-slate-900/95 border-r border-slate-800 flex flex-col h-[calc(100vh-61px)] overflow-y-auto">
      <div className="p-4 space-y-4 flex-1">
        {/* Ingestion Selection */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              1. Aerial Imagery Ingestion
            </span>
            {(selectedFile || selectedPresetId || result) && (
              <button
                onClick={onReset}
                className="text-[11px] text-slate-400 hover:text-slate-200 flex items-center gap-1 transition-colors"
                title="Reset selection"
              >
                <RotateCcw className="h-3 w-3" />
                <span>Reset</span>
              </button>
            )}
          </div>

          {/* Quick Presets */}
          <div className="space-y-1.5">
            <label className="text-[11px] text-slate-400 font-medium">Verified Evaluation Presets:</label>
            <div className="grid grid-cols-1 gap-1.5">
              {presets.map((p) => {
                const isSelected = selectedPresetId === p.id && !selectedFile;
                return (
                  <button
                    key={p.id}
                    onClick={() => onSelectPreset(p.id)}
                    disabled={isProcessing}
                    className={`text-left p-2.5 rounded-lg border text-xs transition-all flex items-start gap-2.5 ${
                      isSelected
                        ? "bg-indigo-950/40 border-indigo-500/60 text-slate-100 shadow-sm"
                        : "bg-slate-800/60 border-slate-700/50 hover:bg-slate-800 text-slate-300 hover:border-slate-600"
                    }`}
                  >
                    <div className="mt-0.5 h-2 w-2 rounded-full bg-indigo-400 flex-shrink-0" />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between gap-1">
                        <span className="font-semibold text-slate-200 truncate">{p.name}</span>
                        <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-900 text-slate-400 border border-slate-700">
                          {p.format}
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400 line-clamp-1 mt-0.5">{p.description}</p>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Dropzone */}
          <div className="pt-1">
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              className="border-2 border-dashed border-slate-700/70 hover:border-slate-500 rounded-xl p-3.5 text-center bg-slate-950/40 transition-colors cursor-pointer group"
              onClick={() => document.getElementById("file-input")?.click()}
            >
              <input
                id="file-input"
                type="file"
                accept=".tif,.tiff,.jpg,.jpeg,.png"
                className="hidden"
                onChange={handleFileChange}
                disabled={isProcessing}
              />
              <UploadCloud className="h-6 w-6 mx-auto text-slate-400 group-hover:text-emerald-400 transition-colors mb-1.5" />
              <p className="text-xs text-slate-300 font-medium">
                Drop custom GeoTIFF / JPG / PNG
              </p>
              <p className="text-[10px] text-slate-400 mt-0.5">
                Supports up to 30cm GSD optical satellite or aerial photography
              </p>
            </div>
          </div>
        </div>

        {/* Selected Image Metadata */}
        {currentImageInfo && (
          <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3 space-y-2 text-xs">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-1.5">
              <span className="text-slate-400 font-medium text-[11px]">Selected Image</span>
              <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                {currentImageInfo.format}
              </span>
            </div>
            <div className="font-mono text-slate-200 text-xs truncate" title={currentImageInfo.name}>
              {currentImageInfo.name}
            </div>
            <div className="grid grid-cols-2 gap-2 text-[11px] pt-1">
              <div>
                <span className="text-slate-400 block">Dimensions:</span>
                <span className="text-slate-200 font-mono">{currentImageInfo.dimensions || "650 x 650"}</span>
              </div>
              <div>
                <span className="text-slate-400 block">Georeferenced:</span>
                <span className={`font-mono ${currentImageInfo.georeferenced ? "text-emerald-400" : "text-amber-400"}`}>
                  {currentImageInfo.georeferenced ? "WGS 84 (CRS84)" : "No (Pixel coords)"}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Primary Action Button */}
        <div>
          <button
            onClick={onRunInference}
            disabled={!currentImageInfo || isProcessing}
            className={`w-full py-2.5 px-4 rounded-xl font-semibold text-xs flex items-center justify-center gap-2 transition-all shadow-md ${
              !currentImageInfo || isProcessing
                ? "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/50"
                : "bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-950/30 active:scale-[0.98]"
            }`}
          >
            {isProcessing ? (
              <>
                <Zap className="h-4 w-4 animate-spin text-emerald-300" />
                <span>Running CPU Inference...</span>
              </>
            ) : (
              <>
                <Sparkles className="h-4 w-4" />
                <span>Extract Buildings (LightUNet)</span>
              </>
            )}
          </button>
        </div>

        {/* Real Processing Stages */}
        {isProcessing && (
          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3 space-y-2 text-xs animate-in fade-in">
            <span className="text-slate-400 font-semibold text-[11px] block uppercase tracking-wider">
              Pipeline Execution Sequence
            </span>
            <div className="space-y-1.5">
              {stages.map((st) => (
                <div key={st.key} className="flex items-center gap-2 text-[11px] text-slate-300">
                  <div className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  <span>{st.label}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Detection Metrics Summary */}
        {result && (
          <div className="bg-slate-950 border border-slate-800/90 rounded-xl p-3.5 space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-emerald-400 flex items-center gap-1.5">
                <CheckCircle2 className="h-3.5 w-3.5" />
                <span>Inference Metrics</span>
              </span>
              <span className="text-[10px] font-mono text-slate-400">
                {result.inference_time_ms} ms
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-slate-900/90 p-2 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Buildings Detected</span>
                <span className="text-lg font-bold text-slate-100 font-mono">{result.building_count}</span>
              </div>
              <div className="bg-slate-900/90 p-2 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Mean Confidence</span>
                <span className="text-lg font-bold text-emerald-400 font-mono">
                  {(result.mean_confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            <div className="space-y-1 text-[11px]">
              <div className="flex justify-between py-0.5">
                <span className="text-slate-400">Model:</span>
                <span className="font-mono text-slate-200">{result.model} ({result.model_checkpoint})</span>
              </div>
              <div className="flex justify-between py-0.5">
                <span className="text-slate-400">Coordinate Space:</span>
                <span className="font-mono text-slate-200">
                  {result.georeferenced ? "Geographic (WGS84)" : "Pixel Coordinates"}
                </span>
              </div>
              <div className="flex justify-between py-0.5">
                <span className="text-slate-400">CRS:</span>
                <span className="font-mono text-slate-200">{result.crs}</span>
              </div>
              {result.ground_truth_metrics && (
                <div className="flex justify-between py-0.5 border-t border-slate-800 pt-1 text-indigo-400">
                  <span>Validation Dice / IoU:</span>
                  <span className="font-mono">
                    {(result.ground_truth_metrics.validation_dice! * 100).toFixed(1)}% / {(result.ground_truth_metrics.validation_iou! * 100).toFixed(1)}%
                  </span>
                </div>
              )}
            </div>

            <button
              onClick={onDownloadGeoJSON}
              className="w-full py-2 px-3 rounded-lg bg-indigo-600/90 hover:bg-indigo-500 text-white text-xs font-medium flex items-center justify-center gap-1.5 transition-colors"
            >
              <Download className="h-3.5 w-3.5" />
              <span>Download GeoJSON</span>
            </button>
          </div>
        )}

        {/* Interactive Building Inspector */}
        {selectedBuilding && (
          <div className="bg-slate-950 border border-emerald-500/40 rounded-xl p-3.5 space-y-2 text-xs animate-in slide-in-from-bottom-2">
            <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
              <span className="font-semibold text-emerald-400 flex items-center gap-1.5">
                <Building2 className="h-3.5 w-3.5" />
                <span>Building Inspector</span>
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-700/50">
                AI DETECTED
              </span>
            </div>

            <div className="space-y-1.5 text-[11px]">
              <div className="flex justify-between">
                <span className="text-slate-400">Feature ID:</span>
                <span className="font-mono text-slate-200 font-bold">{selectedBuilding.properties.id}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Model Source:</span>
                <span className="font-mono text-slate-200">{selectedBuilding.properties.source}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Confidence:</span>
                <span className="font-mono text-emerald-400 font-bold">
                  {(selectedBuilding.properties.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Pixel Area:</span>
                <span className="font-mono text-slate-200">{selectedBuilding.properties.pixel_area} px²</span>
              </div>
              {selectedBuilding.properties.estimated_ground_area_sqm && (
                <div className="flex justify-between">
                  <span className="text-slate-400">Ground Area (Est.):</span>
                  <span className="font-mono text-slate-200 font-bold">
                    {selectedBuilding.properties.estimated_ground_area_sqm} m²
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Footer domain notice */}
      <div className="p-3 border-t border-slate-800 bg-slate-950/80 text-[10px] text-slate-400 flex items-start gap-2">
        <Info className="h-3.5 w-3.5 flex-shrink-0 text-slate-400 mt-0.5" />
        <span>
          Trained on SpaceNet 2 Las Vegas (30cm GSD). Arbitrary external imagery has not been fine-tuned.
        </span>
      </div>
    </aside>
  );
};
