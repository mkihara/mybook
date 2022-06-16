using Pkg

dependencies = [
    "IJulia",
    "Plots",
    "CUDA",
    "BenchmarkTools",
    "Krylov",
    "LinearOperators"
]

Pkg.add(dependencies)
