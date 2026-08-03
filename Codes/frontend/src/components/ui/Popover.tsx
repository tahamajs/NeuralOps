import React, {
  useState,
  createContext,
  useContext,
  useRef,
  useEffect,
} from "react";

const PopoverContext = createContext<{
  open: boolean;
  setOpen: (open: boolean) => void;
  triggerRef: React.RefObject<HTMLDivElement | null>;
} | null>(null);

export function Popover({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <PopoverContext.Provider value={{ open, setOpen, triggerRef }}>
      <div ref={containerRef} className="relative inline-block text-left">
        {children}
      </div>
    </PopoverContext.Provider>
  );
}

export function PopoverTrigger({ children }: { children: React.ReactNode }) {
  const context = useContext(PopoverContext);
  if (!context) throw new Error("PopoverTrigger must be used within Popover");

  return (
    <div
      ref={context.triggerRef}
      onClick={() => context.setOpen(!context.open)}
      className="cursor-pointer"
    >
      {children}
    </div>
  );
}

export function PopoverContent({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  const context = useContext(PopoverContext);
  if (!context) throw new Error("PopoverContent must be used within Popover");

  if (!context.open) return null;

  return (
    <div
      className={`absolute right-0 mt-2 z-50 rounded-md shadow-lg bg-surface border border-slate-700/50 focus:outline-none ${className}`}
    >
      {children}
    </div>
  );
}
