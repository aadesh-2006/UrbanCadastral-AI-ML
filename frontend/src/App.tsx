import React, { useState, useEffect, useRef } from "react";
import { Header } from "./components/Header";
import { Sidebar } from "./components/Sidebar";
import { Workspace } from "./components/Workspace";
import type { InferenceResult, PresetItem, ProcessingStage, ViewMode, GeoJSONFeature } from "./types";

export const App: React.FC = () => {
  const [apiOnline, setApiOnline] = useState(false);
  const [presets, setPresets] = useState<PresetItem[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string | null>("tile_127");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [currentImageInfo, setCurrentImageInfo] = useState<{
    name: string;
    dimensions?: string;
    format: string;
    georeferenced: boolean;
    crs?: string;
  } | null>(null);

  const [processingStage, setProcessingStage] = useState<ProcessingStage>("idle");
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("raw");
  const [selectedBuilding, setSelectedBuilding] = useState<GeoJSONFeature | null>(null);

  // Track whether initial preset has been loaded so we never overwrite user selection
  const initializedRef = useRef(false);

  // Health check heartbeat (every 10s) - ONLY checks connectivity, NEVER resets selection
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch("/api/health");
        setApiOnline(res.ok);
      } catch {
        setApiOnline(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  // Fetch presets ONCE on component mount
  useEffect(() => {
    const loadPresets = async () => {
      try {
        const res = await fetch("/api/presets");
        if (res.ok) {
          const data: PresetItem[] = await res.json();
          setPresets(data);

          // Only initialize default preset once on initial mount
          if (!initializedRef.current) {
            initializedRef.current = true;
            const defaultPreset = data.find((p) => p.id === "tile_127") || data[0];
            if (defaultPreset) {
              setSelectedPresetId(defaultPreset.id);
              setPreviewUrl(defaultPreset.preview_url);
              setCurrentImageInfo({
                name: defaultPreset.name,
                dimensions: defaultPreset.dimensions,
                format: defaultPreset.format,
                georeferenced: defaultPreset.georeferenced,
                crs: defaultPreset.crs
              });
            }
          }
        }
      } catch (err) {
        console.error("Failed to fetch presets", err);
      }
    };

    loadPresets();
  }, []);

  // User explicitly selects a preset
  const handleSelectPreset = (id: string) => {
    const preset = presets.find((p) => p.id === id);
    if (!preset) return;

    setSelectedFile(null);
    setSelectedPresetId(id);
    setPreviewUrl(preset.preview_url);
    setResult(null);
    setSelectedBuilding(null);
    setViewMode("raw");

    setCurrentImageInfo({
      name: preset.name,
      dimensions: preset.dimensions,
      format: preset.format,
      georeferenced: preset.georeferenced,
      crs: preset.crs
    });
  };

  // User explicitly uploads a local file
  const handleFileUpload = (file: File) => {
    setSelectedPresetId(null);
    setSelectedFile(file);
    setResult(null);
    setSelectedBuilding(null);
    setViewMode("raw");

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);

    const ext = file.name.split(".").pop()?.toUpperCase() || "IMAGE";
    const isTiff = ext === "TIF" || ext === "TIFF";

    setCurrentImageInfo({
      name: file.name,
      dimensions: "Pending scan",
      format: isTiff ? "GeoTIFF" : ext,
      georeferenced: isTiff,
      crs: isTiff ? "Embedded Affine/WGS84" : "None"
    });
  };

  // Execute LightUNet inference
  const handleRunInference = async () => {
    setProcessingStage("preprocessing");
    setSelectedBuilding(null);

    // Capture the current image identity before inference
    const activePreset = presets.find((p) => p.id === selectedPresetId);
    const activeName = selectedFile ? selectedFile.name : (activePreset?.name || currentImageInfo?.name || "Ingested Image");
    const activeFormat = selectedFile
      ? (currentImageInfo?.format || "Standard Image")
      : (activePreset?.format || "Standard Image");

    try {
      setTimeout(() => setProcessingStage("segmentation"), 350);
      setTimeout(() => setProcessingStage("cleanup"), 800);
      setTimeout(() => setProcessingStage("polygonization"), 1200);
      setTimeout(() => setProcessingStage("geojson"), 1500);

      let url = "/api/inference";
      const options: RequestInit = { method: "POST" };

      if (selectedFile) {
        const formData = new FormData();
        formData.append("file", selectedFile);
        options.body = formData;
      } else if (selectedPresetId) {
        url += `?preset_id=${selectedPresetId}`;
      } else {
        throw new Error("No image selected");
      }

      const res = await fetch(url, options);
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || "Inference execution failed");
      }

      const data: InferenceResult = await res.json();
      setResult(data);
      setPreviewUrl(data.preview_url);
      setViewMode("overlay");
      setProcessingStage("complete");

      // Preserve the active image selection name and update with actual scanned dimensions
      setCurrentImageInfo({
        name: activeName,
        dimensions: `${data.image_size[1]} x ${data.image_size[0]}`,
        format: activeFormat,
        georeferenced: data.georeferenced,
        crs: data.crs
      });

    } catch (err: any) {
      console.error(err);
      setProcessingStage("error");
      alert(`Inference failed: ${err.message || "Unknown error"}`);
    }
  };

  // Download GeoJSON
  const handleDownloadGeoJSON = () => {
    if (!result?.geojson_data) return;
    const blob = new Blob([JSON.stringify(result.geojson_data, null, 2)], {
      type: "application/geo+json"
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${result.input_image.split(/[\\/]/).pop()?.split(".")[0] || "inference"}_buildings.geojson`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Explicit user reset
  const handleReset = () => {
    setSelectedFile(null);
    setResult(null);
    setSelectedBuilding(null);
    setViewMode("raw");
    if (presets.length > 0) {
      handleSelectPreset(presets[0].id);
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-slate-950 text-slate-100 font-sans">
      <Header apiOnline={apiOnline} />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          presets={presets}
          selectedPresetId={selectedPresetId}
          onSelectPreset={handleSelectPreset}
          onFileUpload={handleFileUpload}
          selectedFile={selectedFile}
          currentImageInfo={currentImageInfo}
          onRunInference={handleRunInference}
          processingStage={processingStage}
          result={result}
          selectedBuilding={selectedBuilding}
          onDownloadGeoJSON={handleDownloadGeoJSON}
          onReset={handleReset}
        />
        <Workspace
          previewUrl={previewUrl}
          result={result}
          viewMode={viewMode}
          onViewModeChange={setViewMode}
          selectedBuilding={selectedBuilding}
          onSelectBuilding={setSelectedBuilding}
        />
      </div>
    </div>
  );
};

export default App;
