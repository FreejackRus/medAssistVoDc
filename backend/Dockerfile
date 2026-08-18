FROM rust:1.87-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
# Create dummy main to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release && rm -rf src

COPY src/ src/
COPY migrations/ migrations/
RUN touch src/main.rs && cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates sqlite3 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/backend ./backend
COPY migrations/ migrations/

RUN mkdir -p storage/uploads storage/exports

EXPOSE 3000
CMD ["./backend"]
