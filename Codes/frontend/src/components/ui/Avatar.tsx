import React from "react";

interface AvatarProps {
  src?: string;
  alt?: string;
  fallback?: string;
  className?: string;
}

export function Avatar({ src, alt, fallback = "?", className = "" }: AvatarProps) {
  return (
    <div className={`relative flex h-10 w-10 shrink-0 overflow-hidden rounded-full bg-slate-800 border border-slate-700 ${className}`}>
      {src ? (
        <img src={src} alt={alt} className="aspect-square h-full w-full object-cover" />
      ) : (
        <div className="flex h-full w-full items-center justify-center rounded-full bg-indigo-500/10 text-indigo-400 font-semibold text-sm">
          {fallback}
        </div>
      )}
    </div>
  );
}
export default Avatar;
