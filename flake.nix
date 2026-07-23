{
  description = "laddu Rust and Python development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          vulkanArch = if system == "x86_64-linux" then "x86_64" else "aarch64";
          python = pkgs.python313;
          nativeLibraries = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.bzip2
            pkgs.libffi
            pkgs.openssl
            pkgs.sqlite
            pkgs.xz
            pkgs.zlib
            pkgs.mpich
            pkgs.vulkan-loader
            pkgs.libGL
            pkgs.libxkbcommon
            pkgs.wayland
          ];
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              cargo
              clippy
              cmake
              just
              mpich
              patchelf
              pkg-config
              prek
              python
              rustc
              rust-analyzer
              rustfmt
              taplo
              uv
              vulkan-loader
              vulkan-tools
            ];

            LD_LIBRARY_PATH = nativeLibraries;
            UV_PYTHON_DOWNLOADS = "never";

            shellHook = ''
              export LADDU_ROOT="$PWD"
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver-32/lib:${nativeLibraries}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              export XDG_DATA_DIRS="/run/opengl-driver/share:${pkgs.vulkan-loader}/share''${XDG_DATA_DIRS:+:$XDG_DATA_DIRS}"
              export WGPU_BACKEND="vulkan"

              nvidia_icd="/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.${vulkanArch}.json"
              if [[ -r "$nvidia_icd" ]]; then
                export VK_DRIVER_FILES="$nvidia_icd"
              fi

              export MPLCONFIGDIR="''${XDG_CACHE_HOME:-$PWD/.cache}/matplotlib"
              mkdir -p "$MPLCONFIGDIR"

              export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
              export MATURIN_PEP517_ARGS="--generate-stubs"
              UV_PYTHON="${python}/bin/python" uv sync --frozen --inexact --no-install-project --project "$PWD/python/laddu"

              export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
              export PATH="$VIRTUAL_ENV/bin:$PATH"
              export PYO3_PYTHON="$VIRTUAL_ENV/bin/python"
              unset PYTHONHOME
            '';
          };
        });

      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt-tree);
    };
}
