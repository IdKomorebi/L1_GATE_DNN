1.gross_actual_interchange_mw：推不准，loss大概0.14，r^2大概0.85

2.metered_load_mw：推的太准了，r^2有0.999，有两个比较极端
prelim_load_avg_hourly和total_gen，验证了一下只需要一个就能推断出来

3.net_actual_interchange_mw:推的也很准，有一个比较极端net_sched_interchange_mw，gate到最后都有0.76

4.marginal_loss_price_rt：推的还可以，0.969，筛选出来有16个Selected，system_energy_price_rt的gate0.767；total_lmp_rt的gate0.72
其中只用最高的1个r^2：（0.91，0.97）
用最高两个r^2有（0.92，0.74）
用最高的三个（0.92，0.76）
用最高的10个（0.96，0.59）
用slected16个（0.96，0.57）

5.congestion_price_da:推的还可以r^2为0.989，有两个一骑绝尘total_lmp_da，system_energy_price_da，gate都在1.4左右
只用最高的一个（0.222，0.737）
最用最高的两个（0.995，0.728）
用selected9个（0.9993，0，6886）

6.total_losses:推的有点不准r^2为0.95，没有特别突出的，有一个稍微领先marginal_loss_price_rt的gate是0.28
只用最高的一个（0.336，0.945）
用最高的三个（0.74，0.945）
用最高的五个（0.87，0.94）
用最高的10个（0.918，0.931）
用selected20个（0.95，0.89）

7.total_lmp_rt：推的挺准，有两个特别高，system_energy_price_rt的gate有1.159，marginal_loss_price_rt有0.819
只用最高的一个（1.0，0.96）
只用这两个特别高的（1.00，0.70）
