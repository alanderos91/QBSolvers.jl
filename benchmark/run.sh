#! /bin/zsh

seed=${1}

n=8192
p=2048
# OLS, n > p
for lambda in 0 1000; do
    for cor in 0.1 0.2 0.4 0.8; do
        echo "OLS ${n}x${p}"
        echo "  rho = ${cor}"
        echo "  lambda = ${lambda}"
        julia OLS.jl ${n} ${p} ${lambda} ${seed} ${cor} > ./OLS/${n}x${p}-lambda=${lambda}-seed=${seed}-cor=${cor}.out
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
    julia OLS.jl ${n} ${p} ${lambda} ${seed} ${cor} > ./OLS/${n}x${p}-lambda=${lambda}-seed=${seed}-cor=${cor}.out
    echo "------------------------------"
    echo
done

# Quantile, n > p
n=8192
p=1024
for cor in 0.0 0.2 0.8; do
    for q in 0.1 0.5 0.9; do
        echo "QREG ${n}x${p}"
        echo "  rho = ${cor}"
        echo "  quantile = ${q}"
        julia QREG.jl ${n} ${p} ${q} ${seed} ${cor} > ./QREG/${n}x${p}-q=${q}-seed=${seed}-cor=${cor}.out
        echo "------------------------------"
        echo
    done
done

