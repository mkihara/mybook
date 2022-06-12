# プログラム言語の比較

読み書きしやすくて、それなりに早いプログラミング言語を求めて、いくつかの言語やライブラリを比較しました。比較したのは、以下の言語です。

* Python
  * NumPy & SciPy
  * JAX
  * CuPy
* Julia

## JAX

[公式のインストール手順にしたがってインストールします。](https://github.com/google/jax#installation)
WindowsでGPUバージョンのJAXを利用する場合、[WSL](https://docs.microsoft.com/ja-jp/windows/wsl/install)で利用するのが良さそうです。
[WSLでCUDAを有効化](https://docs.microsoft.com/ja-jp/windows/ai/directml/gpu-cuda-in-wsl)し、
[CuDNN](https://developer.nvidia.com/cudnn)をインストールします。CuDNNのインストールは、[ここ](https://www.kkaneko.jp/tools/ubuntu/ubuntu_cudnn.html)を参考にしました。
