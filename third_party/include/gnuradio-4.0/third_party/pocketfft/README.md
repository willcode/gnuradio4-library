# PocketFFT

Upstream project: <https://github.com/mreinecke/pocketfft>, `cpp` branch.

Vendored from commit `076cb3d` of 2023-02-14, by way of the distribution package
`pocketfft-devel-1.0^git20230214.076cb3d`. SHA-256 of `pocketfft_hdronly.h` as vendored:
`0eaea21304bf324639ff11022ee63faf48a8fab64afe5c3c2d1efefbe1d7dfa6`. Copied here on 2026-09-06.

The header is taken verbatim and is not reformatted, respelled or otherwise edited; a change here would be an upgrade
to another upstream revision and nothing else. `algorithm/CMakeLists.txt` carries `third_party/include/` as a system
include for the same reason.

The directories above this one mirror the installed layout, so the include path `gnuradio-4.0/third_party/pocketfft/`
is the same in the source tree and in an install. A consumer that finds this project through pkg-config gets only
`-I${includedir}`, and that is all this path needs.

License: BSD-3-Clause, copyright the Max-Planck-Society and Peter Bell, with the odd-length DCT-IV path additionally
copyright Matteo Frigo and the Massachusetts Institute of Technology under the same three clauses. The notice is in
the header itself and in `LICENSE.md`.

Used by `gnuradio-4.0/algorithm/fourier/fft.hpp` as the `FftBackend::PocketFFT` path of `gr::algorithm::FFT`, through
`pocketfft::detail::pocketfft_c` and `pocketfft::detail::pocketfft_r` — the one-dimensional plan objects — and not
through the `c2c`/`r2c` driver, so no plan is shared and the header's global plan cache and its mutex are never
reached.
