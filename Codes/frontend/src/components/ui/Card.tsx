import { cn } from "../../lib/utils";
import { cva } from "class-variance-authority";

const cardVariants = cva("glass transition-all duration-300", {
  variants: {
    padding: {
      sm: "p-3",
      md: "p-5",
      lg: "p-8",
    },
    hover: {
      true: "card-hover-effect",
      false: "",
    },
  },
  defaultVariants: { padding: "md", hover: true },
});

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  padding?: "sm" | "md" | "lg";
  hover?: boolean;
}

export const Card = ({
  children,
  className,
  padding,
  hover,
  ...props
}: CardProps) => (
  <div className={cn(cardVariants({ padding, hover }), className)} {...props}>
    {children}
  </div>
);

export const CardHeader = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("flex flex-col space-y-1.5 p-6", className)} {...props} />
);

export const CardTitle = ({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
  <h3 className={cn("text-2xl font-semibold leading-none tracking-tight", className)} {...props} />
);

export const CardDescription = ({ className, ...props }: React.HTMLAttributes<HTMLParagraphElement>) => (
  <p className={cn("text-sm text-slate-400", className)} {...props} />
);

export const CardContent = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("p-6 pt-0", className)} {...props} />
);
