import React from "react";

interface AvatarProps {
  src?: string;
  alt?: string;
  fallback?: string;
  className?: string;
  size?: string;
}

export function Avatar({
  src,
  alt,
  fallback = "?",
  className = "",
  size = "md",
}: AvatarProps) {
  const sizeClasses: Record<string, string> = {
    sm: "h-8 w-8 text-xs",
    md: "h-10 w-10 text-sm",
    lg: "h-12 w-12 text-base",
  };
  const sizeClass = sizeClasses[size] || "h-10 w-10 text-sm";

  return (
    <div
      className={`relative flex ${sizeClass} shrink-0 overflow-hidden rounded-full bg-slate-800 border border-slate-700 ${className}`}
    >
      {src ? (
        <img
          src={src}
          alt={alt}
          className="aspect-square h-full w-full object-cover"
        />
      ) : (
        <div className="flex h-full w-full items-center justify-center rounded-full bg-indigo-500/10 text-indigo-400 font-semibold">
          {fallback}
        </div>
      )}
    </div>
  );
}

export default Avatar;
