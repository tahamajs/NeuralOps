import { FixedSizeList as List } from "react-window";

interface VirtualizedListProps<T> {
  items: T[];
  height: number;
  itemHeight: number;
  renderItem: (item: T, index: number) => React.ReactNode;
}

export function VirtualizedList<T>({
  items,
  height,
  itemHeight,
  renderItem,
}: VirtualizedListProps<T>) {
  const Row = ({
    index,
    style,
  }: {
    index: number;
    style: React.CSSProperties;
  }) => <div style={style}>{renderItem(items[index], index)}</div>;
  return (
    <List
      height={height}
      itemCount={items.length}
      itemSize={itemHeight}
      width="100%"
    >
      {Row}
    </List>
  );
}
