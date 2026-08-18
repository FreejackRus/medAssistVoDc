{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          # Rust
          pkgs.rustc
          pkgs.cargo
          pkgs.clippy
          pkgs.rustfmt
          pkgs.rust-analyzer

          # Node.js + pnpm
          pkgs.nodejs
          pkgs.pnpm

          # Python
          pkgs.python313
          pkgs.uv

          # System deps for Rust crates (reqwest, sqlx, etc.)
          pkgs.pkg-config
          pkgs.openssl

          # SQLite for sqlx
          pkgs.sqlite

          # LLM runtime
          pkgs.ollama
        ];

        env = {
          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
          UV_PYTHON = "${pkgs.python313}/bin/python3";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
          ];
        };

        shellHook = ''
          echo "medAssistant dev shell"
          echo "  Rust:   $(rustc --version)"
          echo "  Node:   $(node --version)"
          echo "  pnpm:   $(pnpm --version)"
          echo "  Python: $(python3 --version)"
          echo "  uv:     $(uv --version)"
        '';
      };
    };
}
