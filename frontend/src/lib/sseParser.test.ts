import { describe, expect, it } from "vitest";
import { parseSseStream } from "@/lib/sseParser";

function streamFromChunks(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)));
      controller.close();
    },
  });
}

describe("parseSseStream", () => {
  it("parses chunked CRLF frames with ids and multiline data", async () => {
    const stream = streamFromChunks([
      "id: 4\r\nevent: tok",
      "en\r\ndata: first\r\ndata: second\r\n\r\n",
      ": keepalive\n\ndata: final",
    ]);

    const frames = [];
    for await (const frame of parseSseStream(stream, "message")) {
      frames.push(frame);
    }

    expect(frames).toEqual([
      { id: 4, event: "token", data: "first\nsecond" },
      { id: 0, event: "message", data: "final" },
    ]);
  });

  it("resets event and id after each frame", async () => {
    const stream = streamFromChunks([
      "id: 9\nevent: done\ndata: \"ok\"\n\ndata: \"next\"\n\n",
    ]);

    const frames = [];
    for await (const frame of parseSseStream(stream, "token")) {
      frames.push(frame);
    }

    expect(frames[1]).toEqual({ id: 0, event: "token", data: "\"next\"" });
  });
});
