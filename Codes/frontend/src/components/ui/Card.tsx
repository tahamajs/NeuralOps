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
