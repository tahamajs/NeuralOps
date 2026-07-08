import { useState, createContext, useContext } from "react";
import { cn } from "../../lib/utils";

interface TabsContextValue {
  active: string;
  setActive: (val: string) => void;
}

const TabsContext = createContext<TabsContextValue>({
  active: "",
  setActive: () => {},
});

export const Tabs = ({
  defaultValue,
  value,
  onValueChange,
  children,
  className,
}: {
  defaultValue?: string;
  value?: string;
  onValueChange?: (val: string) => void;
  children: React.ReactNode;
  className?: string;
}) => {
  const [localActive, setLocalActive] = useState(defaultValue || "");
  const active = value !== undefined ? value : localActive;
  const setActive = (val: string) => {
    if (onValueChange) onValueChange(val);
    setLocalActive(val);
  };
  return (
    <TabsContext.Provider value={{ active, setActive }}>
      <div className={cn("flex flex-col", className)}>{children}</div>
    </TabsContext.Provider>
  );
};

export const TabList = ({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) => (
  <div className={cn("flex gap-1 p-1 bg-white/5 rounded-xl", className)}>
    {children}
  </div>
);

export const Tab = ({
  value,
  children,
}: {
  value: string;
  children: React.ReactNode;
}) => {
  const { active, setActive } = useContext(TabsContext);
  return (
    <button
      onClick={() => setActive(value)}
      className={cn(
        "px-4 py-2 rounded-lg text-sm font-semibold transition-all",
        active === value
          ? "bg-neon-blue text-white shadow-lg"
          : "text-slate-400 hover:text-white hover:bg-white/5",
      )}
    >
      {children}
    </button>
  );
};

export const TabPanel = ({
  value,
  children,
  className,
}: {
  value: string;
  children: React.ReactNode;
  className?: string;
}) => {
  const { active } = useContext(TabsContext);
  if (active !== value) return null;
  return <div className={cn("pt-4", className)}>{children}</div>;
};
