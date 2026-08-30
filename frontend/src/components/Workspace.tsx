import React, { useState, useRef, useEffect } from "react";
import {
  ZoomIn,
  ZoomOut,
  Maximize,
  Layers,
  Sliders,
  Compass,
  AlertTriangle,
  CheckCircle,
  Tag
} from "lucide-react";
import type { InferenceResult, ViewMode, GeoJSONFeature } from "../types";

interface WorkspaceProps {
  previewUrl: string | null;
  result: InferenceResult | null;
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
  selectedBuilding: GeoJSONFeature | null;
  onSelectBuilding: (feature: GeoJSONFeature | null) => void;
}

export const Workspace: React.FC<WorkspaceProps> = ({
  previewUrl,
  result,
  viewMode,
  onViewModeChange,
  selectedBuilding,
  onSelectBuilding
}) => {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [opacity, setOpacity] = useState(0.85);
  const [showLabels, setShowLabels] = useState(true);

  const containerRef = useRef<HTMLDivElement>(null);

  // Reset zoom & pan when image changes
  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [previewUrl, result?.input_image]);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y
    });
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.15 : 0.15;
    setZoom((prev) => Math.min(Math.max(0.5, prev + delta), 4.0));
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // Image source to display based on view mode
  const getDisplayImageSrc = () => {
    if (!result && previewUrl) return previewUrl;
    if (!result) return null;

    if (viewMode === "mask") {
      return result.mask_url;
    }
    if (viewMode === "overlay") {
      return result.overlay_url;
    }
    return result.preview_url || previewUrl;
  };

  const currentSrc = getDisplayImageSrc();
  const imgWidth = result?.image_size[1] || 650;
  const imgHeight = result?.image_size[0] || 650;

  return (
    <div className="flex-1 flex flex-col h-[calc(100vh-61px)] bg-slate-950 relative overflow-hidden select-none">
      {/* Top Workspace Toolbar */}
      <div className="h-12 bg-slate-900/90 border-b border-slate-800 px-4 flex items-center justify-between z-10 gap-2 flex-wrap">
        {/* View Mode Controls */}
        <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
          <button
            onClick={() => onViewModeChange("raw")}
            className={`px-3 py-1 rounded-md font-medium transition-colors ${
              viewMode === "raw"
                ? "bg-slate-800 text-slate-100 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            RAW IMAGE
          </button>
          <button
            onClick={() => onViewModeChange("mask")}
            disabled={!result}
            className={`px-3 py-1 rounded-md font-medium transition-colors ${
              !result
                ? "text-slate-600 cursor-not-allowed"
                : viewMode === "mask"
                ? "bg-rose-950/80 text-rose-300 border border-rose-700/50 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            AI BUILDING MASK
          </button>
          <button
            onClick={() => onViewModeChange("overlay")}
            disabled={!result}
            className={`px-3 py-1 rounded-md font-medium transition-colors ${
              !result
                ? "text-slate-600 cursor-not-allowed"
                : viewMode === "overlay"
                ? "bg-emerald-950/80 text-emerald-300 border border-emerald-700/50 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            RAW + AI OVERLAY
          </button>
        </div>

        {/* Overlay & Mask Controls */}
        {result && (viewMode === "overlay" || viewMode === "mask") && (
          <div className="flex items-center gap-3 bg-slate-950 px-3 py-1 rounded-lg border border-slate-800 text-xs">
            {viewMode === "overlay" && (
              <div className="flex items-center gap-1.5 text-slate-400">
                <Sliders className="h-3.5 w-3.5" />
                <span className="text-[11px]">Opacity:</span>
                <input
                  type="range"
                  min="0.2"
                  max="1"
                  step="0.05"
                  value={opacity}
                  onChange={(e) => setOpacity(parseFloat(e.target.value))}
                  className="w-16 accent-emerald-500 h-1.5 rounded-lg cursor-pointer"
                />
                <span className="font-mono text-[10px] w-7 text-slate-300">
                  {Math.round(opacity * 100)}%
                </span>
              </div>
            )}

            {viewMode === "mask" && (
              <div className="flex items-center gap-1.5 text-slate-400">
                <Layers className="h-3.5 w-3.5 text-rose-400" />
                <span className="text-[11px] font-medium text-slate-300">Semantic Cadastral View</span>
              </div>
            )}

            <button
              onClick={() => setShowLabels(!showLabels)}
              className={`flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] font-medium transition-colors ${
                showLabels
                  ? viewMode === "mask"
                    ? "text-rose-400 bg-rose-950/50"
                    : "text-emerald-400 bg-emerald-950/50"
                  : "text-slate-500 hover:text-slate-300"
              }`}
              title="Toggle Footprint Labels"
            >
              <Tag className="h-3 w-3" />
              <span>{viewMode === "mask" ? "Footprint IDs" : "Labels"}</span>
            </button>
          </div>
        )}

        {/* View & Georeferencing Status Callout */}
        <div className="hidden lg:flex items-center gap-2 text-xs">
          {result && (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60 text-slate-300">
              <Compass className="h-3.5 w-3.5 text-emerald-400" />
              <span className="font-mono text-slate-400">CRS:</span>
              <span className="font-medium text-slate-200">{result.crs}</span>
              <span className={`ml-1 text-[10px] px-1 py-0.2 rounded font-mono ${
                result.georeferenced ? "bg-emerald-950 text-emerald-400" : "bg-amber-950 text-amber-400"
              }`}>
                {result.georeferenced ? "Georeferenced" : "Image Coords"}
              </span>
            </div>
          )}
        </div>

        {/* Zoom & Reset Controls */}
        <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
          <button
            onClick={() => setZoom((z) => Math.max(0.5, z - 0.2))}
            className="p-1.5 hover:bg-slate-800 text-slate-300 rounded"
            title="Zoom Out"
          >
            <ZoomOut className="h-3.5 w-3.5" />
          </button>
          <span className="px-2 font-mono text-[11px] text-slate-300 min-w-[45px] text-center">
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={() => setZoom((z) => Math.min(4.0, z + 0.2))}
            className="p-1.5 hover:bg-slate-800 text-slate-300 rounded"
            title="Zoom In"
          >
            <ZoomIn className="h-3.5 w-3.5" />
          </button>
          <div className="h-4 w-px bg-slate-800 mx-1" />
          <button
            onClick={resetView}
            className="p-1.5 hover:bg-slate-800 text-slate-300 rounded"
            title="Reset Pan & Zoom"
          >
            <Maximize className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Hidden SVG Filter Definition for GIS Cadastral Semantic Mask Colorization */}
      <svg className="absolute w-0 h-0 pointer-events-none opacity-0 overflow-hidden" aria-hidden="true">
        <defs>
          <filter id="gis-cadastral-mask" colorInterpolationFilters="sRGB">
            <feColorMatrix
              type="matrix"
              values="
                0.92 0 0 0 0.04
                0.20 0 0 0 0.05
                0.34 0 0 0 0.08
                0    0 0 1 0
              "
            />
          </filter>
        </defs>
      </svg>

      {/* Main Viewport Container */}
      <div
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        className={`flex-1 flex items-center justify-center p-6 cursor-${isDragging ? "grabbing" : "grab"} relative overflow-hidden`}
      >
        {currentSrc ? (
          <div
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: "center center",
              transition: isDragging ? "none" : "transform 0.08s ease-out"
            }}
            className="relative select-none shadow-2xl rounded-lg border border-slate-800/80 bg-slate-900 overflow-hidden"
          >
            <img
              src={currentSrc}
              alt="Aerial Ingestion View"
              style={{
                opacity: viewMode === "overlay" ? opacity : 1.0,
                filter: viewMode === "mask" ? "url(#gis-cadastral-mask)" : "none"
              }}
              className="max-h-[75vh] w-auto object-contain rounded-lg pointer-events-none transition-opacity duration-150"
              draggable={false}
            />

            {/* Interactive SVG Polygons Layer (Active in both Overlay and Semantic Mask Views) */}
            {result?.polygons_pixel && (viewMode === "overlay" || viewMode === "mask") && (
              <svg
                viewBox={`0 0 ${imgWidth} ${imgHeight}`}
                className="absolute inset-0 w-full h-full pointer-events-auto"
                style={{ overflow: "visible" }}
              >
                {result.polygons_pixel.map((poly) => {
                  const isSelected = selectedBuilding?.id === poly.id;
                  const ptsString = poly.points.map((pt) => `${pt[0]},${pt[1]}`).join(" ");

                  // Calculate center for label
                  const avgX = poly.points.reduce((acc, p) => acc + p[0], 0) / poly.points.length;
                  const avgY = poly.points.reduce((acc, p) => acc + p[1], 0) / poly.points.length;

                  const isMaskMode = viewMode === "mask";
                  const defaultFill = isMaskMode ? "rgba(244, 63, 94, 0.45)" : "transparent";
                  const defaultStroke = isMaskMode ? "#fb7185" : "transparent";
                  const strokeWidth = isSelected ? 2.5 : (isMaskMode ? 1.5 : 1.5);

                  return (
                    <g key={poly.id} className="cursor-pointer group">
                      <polygon
                        points={ptsString}
                        onClick={(e) => {
                          e.stopPropagation();
                          const feat = result.geojson_data.features.find((f) => f.id === poly.id) || {
                            type: "Feature",
                            id: poly.id,
                            geometry: { type: "Polygon", coordinates: [] },
                            properties: poly.properties
                          };
                          onSelectBuilding(feat);
                        }}
                        fill={isSelected ? "rgba(6, 182, 212, 0.55)" : defaultFill}
                        stroke={isSelected ? "#38bdf8" : defaultStroke}
                        strokeWidth={strokeWidth}
                        className={`transition-all ${
                          isMaskMode
                            ? "hover:fill-rose-500/70 hover:stroke-rose-200"
                            : "hover:fill-cyan-500/30 hover:stroke-cyan-400"
                        }`}
                      />
                      {showLabels && (
                        <text
                          x={avgX}
                          y={avgY}
                          textAnchor="middle"
                          dominantBaseline="central"
                          fill={isSelected ? "#38bdf8" : "#ffffff"}
                          fontSize={isMaskMode ? "10" : "11"}
                          fontWeight="bold"
                          className="pointer-events-none drop-shadow-md select-none font-mono"
                        >
                          {isMaskMode ? poly.id.replace("BLD-", "#") : `${Math.round(poly.confidence * 100)}%`}
                        </text>
                      )}
                    </g>
                  );
                })}
              </svg>
            )}
          </div>
        ) : (
          <div className="text-center text-slate-500 space-y-2">
            <div className="h-12 w-12 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center mx-auto text-slate-400">
              <Layers className="h-6 w-6" />
            </div>
            <p className="text-sm font-medium text-slate-400">No aerial imagery ingested</p>
            <p className="text-xs text-slate-400 max-w-xs mx-auto">
              Select one of the verified SpaceNet presets or upload a custom aerial GeoTIFF/JPG image.
            </p>
          </div>
        )}

        {/* Informational overlay banner for plain images */}
        {result && !result.georeferenced && (
          <div className="absolute bottom-4 left-6 z-10 bg-slate-900/90 border border-amber-500/40 px-3 py-1.5 rounded-lg text-xs text-amber-300 flex items-center gap-2 shadow-lg">
            <AlertTriangle className="h-3.5 w-3.5 flex-shrink-0" />
            <span>Image is not georeferenced. Building polygons are shown in local image coordinates.</span>
          </div>
        )}

        {result && result.georeferenced && (
          <div className="absolute bottom-4 left-6 z-10 bg-slate-900/90 border border-emerald-500/40 px-3 py-1.5 rounded-lg text-xs text-emerald-300 flex items-center gap-2 shadow-lg">
            <CheckCircle className="h-3.5 w-3.5 flex-shrink-0" />
            <span>Georeferenced SpaceNet imagery (EPSG:4326). Geospatial affine transform preserved.</span>
          </div>
        )}

        {/* GIS Semantic Cadastral Legend for Mask Mode */}
        {result && viewMode === "mask" && (
          <div className="absolute bottom-4 right-6 z-10 bg-slate-900/95 backdrop-blur-md border border-slate-800 p-3 rounded-xl shadow-2xl text-xs space-y-2 max-w-[260px] animate-in fade-in select-none">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-1.5">
              <div className="flex items-center gap-1.5">
                <Layers className="h-3.5 w-3.5 text-rose-400" />
                <span className="font-semibold text-slate-200 text-[11px] tracking-wide uppercase">
                  Cadastral Mask Legend
                </span>
              </div>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                LightUNet
              </span>
            </div>

            <div className="space-y-1.5 text-[11px]">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <span className="w-3.5 h-3.5 rounded-sm bg-rose-500/80 border border-rose-400 flex-shrink-0 shadow-sm shadow-rose-950" />
                  <span className="text-slate-200 font-medium">Building Footprint</span>
                </div>
                <span className="font-mono text-[10px] text-rose-300 font-semibold">{result.building_count} Detected</span>
              </div>
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <span className="w-3.5 h-3.5 rounded-sm bg-slate-950 border border-slate-800 flex-shrink-0" />
                  <span className="text-slate-400">Background / Non-Built</span>
                </div>
                <span className="font-mono text-[10px] text-slate-500">Unclassified</span>
              </div>
            </div>

            <div className="pt-1.5 border-t border-slate-800/60 text-[10px] text-slate-400 flex items-center justify-between font-mono">
              <span>Class: Building (Binary)</span>
              <span className="text-emerald-400 font-semibold">{Math.round(result.mean_confidence * 100)}% Conf</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
