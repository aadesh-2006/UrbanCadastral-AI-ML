export interface GeoJSONFeatureProperties {
  id: string;
  class: string;
  source: string;
  confidence: number;
  pixel_area?: number;
  estimated_ground_area_sqm?: number;
  georeferenced: boolean;
  model_checkpoint?: string;
  tile_id?: string;
}

export interface GeoJSONFeature {
  type: "Feature";
  id: string;
  geometry: {
    type: "Polygon" | "MultiPolygon";
    coordinates: number[][][] | number[][][][];
  };
  properties: GeoJSONFeatureProperties;
}

export interface GeoJSONData {
  type: "FeatureCollection";
  name?: string;
  crs?: {
    type: string;
    properties: { name: string };
  };
  properties?: Record<string, any>;
  features: GeoJSONFeature[];
}

export interface InferenceResult {
  input_image: string;
  model: string;
  model_checkpoint: string;
  image_size: [number, number, number];
  building_count: number;
  mean_confidence: number;
  inference_time_ms: number;
  georeferenced: boolean;
  crs: string;
  preview_url: string;
  mask_url: string;
  overlay_url: string;
  geojson_url: string;
  geojson_data: GeoJSONData;
  polygons_pixel?: {
    id: string;
    confidence: number;
    points: [number, number][];
    properties: GeoJSONFeatureProperties;
  }[];
  ground_truth_metrics?: {
    validation_dice?: number;
    validation_iou?: number;
  };
}

export interface PresetItem {
  id: string;
  name: string;
  description: string;
  format: string;
  dimensions: string;
  georeferenced: boolean;
  crs: string;
  preview_url: string;
}

export type ViewMode = "raw" | "mask" | "overlay";

export type ProcessingStage =
  | "idle"
  | "preprocessing"
  | "segmentation"
  | "cleanup"
  | "polygonization"
  | "geojson"
  | "complete"
  | "error";
