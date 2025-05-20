{ pkgs }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Python and tools
    python39
    python39Packages.pip
    python39Packages.virtualenv
    python39Packages.setuptools
    pkgs.python310Full
    pkgs.python310Packages.pip
    pkgs.python310Packages.setuptools
    pkgs.python310Packages.tensorflow
    
    # System dependencies
    ffmpeg
    libwebp
    zlib
    libjpeg
    libpng
    libtiff
    openjpeg
    tcl
    tk
    xorg.libX11
    xorg.libXext
    xorg.libXrender
    xorg.libXtst
    xorg.libXi
    xorg.libXft
    libGL
    libGLU
    glib
    cairo
    pango
    gdk-pixbuf
    atk
    gtk3
    
    # Development tools
    git
    vim
    wget
    curl
  ];

  # Environment variables
  PYTHONPATH = ".";
  PYTHONIOENCODING = "utf-8";
  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.libjpeg}/lib:${pkgs.libpng}/lib:${pkgs.libtiff}/lib:${pkgs.openjpeg}/lib:${pkgs.tcl}/lib:${pkgs.tk}/lib:${pkgs.xorg.libX11}/lib:${pkgs.xorg.libXext}/lib:${pkgs.xorg.libXrender}/lib:${pkgs.xorg.libXtst}/lib:${pkgs.xorg.libXi}/lib:${pkgs.xorg.libXft}/lib:${pkgs.libGL}/lib:${pkgs.libGLU}/lib:${pkgs.glib}/lib:${pkgs.cairo}/lib:${pkgs.pango}/lib:${pkgs.gdk-pixbuf}/lib:${pkgs.atk}/lib:${pkgs.gtk3}/lib";
  
  # Fix for some Python packages that expect these paths
  TCL_LIBRARY = "${pkgs.tcl}/lib";
  TK_LIBRARY = "${pkgs.tk}/lib";
  
  # Fix for OpenCV
  OPENCV_LOG_LEVEL = "OFF";
  
  # Fix for TensorFlow
  TF_CPP_MIN_LOG_LEVEL = "3";
}
