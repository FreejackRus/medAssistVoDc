interface TextSmootherOptions {
  initialText?: string;
  onText: (text: string) => void;
}

export interface TextSmoother {
  enqueue(text: string): void;
  flush(): void;
  getText(): string;
  stop(): void;
}

function takeSize(backlog: number): number {
  if (backlog > 3000) return 160;
  if (backlog > 1200) return 80;
  if (backlog > 400) return 32;
  if (backlog > 120) return 12;
  return 4;
}

export function createTextSmoother({
  initialText = "",
  onText,
}: TextSmootherOptions): TextSmoother {
  let displayed = initialText;
  let pending = "";
  let frame: number | null = null;
  let stopped = false;

  const cancelFrame = () => {
    if (frame !== null) {
      cancelAnimationFrame(frame);
      frame = null;
    }
  };

  const step = () => {
    frame = null;
    if (stopped || pending.length === 0) return;

    const size = takeSize(pending.length);
    displayed += pending.slice(0, size);
    pending = pending.slice(size);
    onText(displayed);

    if (pending.length > 0) {
      frame = requestAnimationFrame(step);
    }
  };

  const schedule = () => {
    if (!stopped && frame === null && pending.length > 0) {
      frame = requestAnimationFrame(step);
    }
  };

  return {
    enqueue(text: string) {
      if (stopped || text.length === 0) return;
      pending += text;
      schedule();
    },
    flush() {
      if (stopped) return;
      cancelFrame();
      if (pending.length > 0) {
        displayed += pending;
        pending = "";
        onText(displayed);
      }
    },
    getText() {
      return displayed + pending;
    },
    stop() {
      stopped = true;
      pending = "";
      cancelFrame();
    },
  };
}
