#! /bin/zsh

seed=${1}

#
# TABLE 1: Constrained OLS
#
julia -t auto OLSConstrained.jl > ./OLS/seed=${seed}.out

#
# TABLE 2: Quantile Regression, n > p
#
n=8192
p=1024
for cor in 0.0 0.2 0.8; do
    # for q in 0.1 0.5 0.9; do
    for q in 0.5; do
        echo "QREG ${n}x${p}"
        echo "  rho = ${cor}"
        echo "  quantile = ${q}"
        julia -t auto QREG.jl ${n} ${p} ${q} ${seed} ${cor} > ./QREG/${n}x${p}-q=${q}-seed=${seed}-cor=${cor}.out
        echo "------------------------------"
        echo
    done
done

#
# TABLE 3: Markowitz Portfolio Optimization
#
julia -t auto Mean-Variance_Portfolio.jl > ./Portfolio/seed=${seed}.out

#
# TABLE 4: NNQP
#
julia -t auto Non-Negative_Quadratic_Programming.jl > ./NNQP/seed=${seed}.out

#
# TABLE 5: LASSO
#
julia -t auto LASSO.jl > ./LASSO/seed=${seed}.out

#
# TABLE 6: Fused LASSO Proximity
#
julia -t auto TV.jl > ./TV/seed=${seed}.out

#
# Extra Benchmarks
#
n=8192
p=2048
# OLS, n > p
for lambda in 0 1000; do
    for cor in 0.1 0.2 0.4 0.8; do
        echo "OLS ${n}x${p}"
        echo "  rho = ${cor}"
        echo "  lambda = ${lambda}"
        julia -t auto OLS.jl ${n} ${p} ${lambda} ${seed} ${cor} > ./OLS/${n}x${p}-lambda=${lambda}-seed=${seed}-cor=${cor}.out
        echo "------------------------------"
        echo
    done
done

# OLS + Ridge, n < p
n=1024
p=8192
lambda=1000
for cor in 0.1 0.2 0.4 0.8; do
    echo "OLS ${n}x${p}"
    echo "  rho = ${cor}"
    echo "  lambda = ${lambda}"
    julia -t auto OLS.jl ${n} ${p} ${lambda} ${seed} ${cor} > ./OLS/${n}x${p}-lambda=${lambda}-seed=${seed}-cor=${cor}.out
    echo "------------------------------"
    echo
done

