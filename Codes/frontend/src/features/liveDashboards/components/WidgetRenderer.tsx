import { MetricWidget } from "./MetricWidget";
import { LineChartWidget } from "./LineChartWidget";
import { TableWidget } from "./TableWidget";
import { LogWidget } from "./LogWidget";
import { ScatterWidget } from "./ScatterWidget";
import { Widget } from "../types";

export function WidgetRenderer({
  widget,
  data,
}: {
  widget: Widget;
  data: any[];
}) {
  switch (widget.type) {
    case "metric":
      return (
        <MetricWidget
          data={data}
          metricKey={widget.data_source.metric_key}
          config={widget.config}
        />
      );
    case "line_chart":
      return <LineChartWidget data={data} config={widget.config} />;
    case "table":
      return <TableWidget data={data} config={widget.config} />;
    case "log":
      return <LogWidget data={data} />;
    case "scatter":
      return <ScatterWidget data={data} config={widget.config} />;
    default:
      return (
        <div className="text-slate-400 text-sm">Unsupported widget type</div>
      );
  }
}
