import Markdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const REHYPE_PLUGINS: any[] = [rehypeHighlight];
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const REMARK_PLUGINS: any[] = [remarkGfm];

interface Props {
  content: string;
  className?: string;
  variant?: "default" | "algorithm";
}

export default function MarkdownRenderer({ content, className, variant = "default" }: Props) {
  return (
    <div
      className={cn(
        "prose prose-sm max-w-none dark:prose-invert",
        variant === "algorithm" && "algorithm-markdown",
        className,
      )}
    >
      <Markdown
        rehypePlugins={REHYPE_PLUGINS}
        remarkPlugins={REMARK_PLUGINS}
      >
        {content}
      </Markdown>
    </div>
  );
}
