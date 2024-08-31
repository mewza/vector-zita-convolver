SIMD Vectorized Zita Convolver v1.1
------------------------------------------------------------------------------------------

1.1 UPDATE: I decided to completely strip away non-vector (seriealized) code and separate only
the vectorized version. For serial implementation look on GITHUB for ZitaConvolver. I discovered
there may be an issue with concolving stereo mapping, speking of which I will also include a C++
top level class how to deal with Vectorized Zita code in your project. It is coming...

And here it is! As promised... This is SIMD optimzized (vectorized) templated C++ version of 
the latest ZitaConvolver.  It utilizes AVFFT SIMD vector optimized FFT for best performance, 
there may be bugs still use at your own risk, or fix them and send me fixes.. but it works and 
it doesn't crash. What else... IT is FASTer than PFFFT'd ZitaConvolver.

Only Use technology for benefit of humanity never to enslave it!
