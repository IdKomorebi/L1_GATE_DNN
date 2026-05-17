1.metered_load_mw：推的太准了，r^2有0.999，有两个比较极端
prelim_load_avg_hourly和total_gen，验证了一下只需要一个就能推断出来
1*.metered_load_mw:没啥说法，拿前几个，后几个都能推，可以设置最高两个来推可能还有点效果


2.net_actual_interchange_mw:典型的推的很准且有一个极端net_sched_interchange_mw
  net_sched_interchange_mw：几乎一模一样，这两个互为对方极端，两个绑定
2*.net_actual_interchange_mw且排除net_sched_interchange_mw：推的也还行，r^2可能有0.993，四个比较高
用selected的4个（0.999,0.93）
用最高的前6个（0.999，0.92）
用最高的10个（0.999,0.92）
用最高的20个（0.999,0.902）

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

5*一旦去掉这两个就不准了，r^2变成0.7了
只去掉total_lmp_da的话也不准,r^2变成0.76左右
随意还是还是推荐设置最高两个来推

6.total_losses:推的有点不准r^2为0.95，没有特别突出的，有一个稍微领先marginal_loss_price_rt的gate是0.28
只用最高的一个（0.336，0.945）
用最高的三个（0.74，0.945）
用最高的五个（0.87，0.94）
用最高的10个（0.918，0.931）
用selected20个（0.95，0.89）

7.total_lmp_rt：推的挺准，有两个特别高，system_energy_price_rt的gate有1.159，marginal_loss_price_rt有0.819
只用最高的一个（1.0，0.96）
只用这两个特别高的（1.00，0.70）

7*.去掉这两个，r^2变成0.9，推荐n=7-15


8*.gross_actual_interchange_mw:推的还行，loss在0.12左右，r^2在0.87左右
只用最高的两个（0.66,0.68）
只用最高的五个（0.74,0.71）
只用最高的10个（0.852,0.666）
用selected的32个（0.877,0.526）

