{ pkgs }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python39
    python39Packages.pip
    python39Packages.virtualenv
    # Add any other system dependencies here
  ];

  # Environment variables
  PYTHONPATH = ".";
  PYTHONIOENCODING = "utf-8";
  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
}
