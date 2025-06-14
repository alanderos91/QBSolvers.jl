#! /bin/zsh

use_noise=${1}
ngroups=${2}
seed=1903

n=8192
p=2048
# OLS, n > p
for lambda in 0 1000; do
    for cor in indep diag blkdiag band blkband; do
        echo "OLS ${n}x${p}"
        echo "lambda = ${lambda}"
        echo "${cor} ${use_noise} ${ngroups}"
        julia OLS.jl ${n} ${p} ${lambda} ${seed} ${cor} ${use_noise} ${ngroups} > ./OLS/${n}x${p}-lambda=${lambda}-seed=${seed}-cor=${cor}-noisy=${use_noise}-ngroups=${ngroups}.out
        echo "------------------------------"
        echo
    done
done

# OLS + Ridge, n < p
n=1024
p=8192
lambda=1000
for cor in indep diag blkdiag band blkband; do
    echo "OLS ${n}x${p}"
    echo "lambda = ${lambda}"
    echo "${cor} ${use_noise} ${ngroups}"
    julia OLS.jl ${n} ${p} ${lambda} ${seed} ${cor} ${use_noise} ${ngroups} > ./OLS/${n}x${p}-lambda=${lambda}-seed=${seed}-cor=${cor}-noisy=${use_noise}-ngroups=${ngroups}.out
    echo "------------------------------"
    echo
done

# Quantile, n > p
n=8192
p=1023
q=0.5
for cor in indep diag blkdiag band blkband; do
    echo "QREG ${n}x${p}"
    echo "quantile = ${q}"
    echo "${cor} ${use_noise} ${ngroups}"
    julia QREG.jl ${n} ${p} ${q} ${seed} ${cor} ${use_noise} ${ngroups} > ./QREG/${n}x${p}-q=${q}-seed=${seed}-cor=${cor}-noisy=${use_noise}-ngroups=${ngroups}.out
    echo "------------------------------"
    echo
done
