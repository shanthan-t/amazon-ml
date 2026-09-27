# ICU 77.1 Windows runtime

`setup_windows.ps1` downloads the official ICU4C 77.1 x64 Windows binary archive and installs only `icudt77.dll`, `icuuc77.dll`, and `icuin77.dll` into this directory. The Python binding calls the ICU C API directly and checks that the runtime reports version 77.1.0.0. No substitute transliterator is allowed. The ICU license is included here.
