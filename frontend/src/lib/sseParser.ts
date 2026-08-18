export interface SseFrame {
  id: number;
  event: string;
  data: string;
}

export async function* parseSseStream(
  stream: ReadableStream<Uint8Array>,
  defaultEvent: string,
): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let currentEvent = defaultEvent;
  let currentId = 0;
  let dataLines: string[] = [];

  const dispatch = (): SseFrame | null => {
    if (dataLines.length === 0) {
      currentEvent = defaultEvent;
      currentId = 0;
      return null;
    }

    const frame = {
      id: currentId,
      event: currentEvent,
      data: dataLines.join("\n"),
    };
    currentEvent = defaultEvent;
    currentId = 0;
    dataLines = [];
    return frame;
  };

  const handleLine = (rawLine: string): SseFrame | null => {
    const line = rawLine.replace(/\r$/, "");
    if (line === "") return dispatch();
    if (line.startsWith(":")) return null;
    if (line.startsWith("id:")) {
      const parsed = Number.parseInt(line.slice(3).trim(), 10);
      currentId = Number.isFinite(parsed) ? parsed : 0;
      return null;
    }
    if (line.startsWith("event:")) {
      currentEvent = line.slice(6).trim() || defaultEvent;
      return null;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).replace(/^ /, ""));
    }
    return null;
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const frame = handleLine(line);
        if (frame) yield frame;
      }
    }

    buffer += decoder.decode();
    if (buffer.length > 0) {
      const frame = handleLine(buffer);
      if (frame) yield frame;
    }
    const finalFrame = dispatch();
    if (finalFrame) yield finalFrame;
  } finally {
    reader.releaseLock();
  }
}
