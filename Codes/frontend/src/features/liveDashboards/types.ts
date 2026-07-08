export type WidgetType =
  | "metric"
  | "line_chart"
  | "scatter"
  | "table"
  | "log"
  | "parallel_coords"
  | "gauge";
export type DataSourceType = "experiment" | "run" | "system" | "custom";

export interface Dashboard {
  id: string;
  name: string;
  description?: string;
  is_public: boolean;
  refresh_interval: number; // seconds, 0 = disabled
  widgets: Widget[];
  created_at: string;
  updated_at: string;
}

export interface Widget {
  id: string;
  type: WidgetType;
  title: string;
  data_source: {
    type: DataSourceType;
    id?: string;
    query?: string;
    metric_key?: string;
    params?: Record<string, any>;
  };
  position: { x: number; y: number; w: number; h: number };
  config?: Record<string, any>; // e.g., color, y-axis range, aggregation
}

export interface WidgetDefinition {
  type: WidgetType;
  name: string;
  icon: string;
  defaultWidth: number;
  defaultHeight: number;
  description: string;
}

export interface LiveDataPoint {
  timestamp: string;
  value: number;
  metric?: string;
  [key: string]: any;
}
