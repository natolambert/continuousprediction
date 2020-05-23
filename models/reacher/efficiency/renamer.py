import os

for type in ['d', 't', 'de', 'te', 'p', 'tp', 'pe', 'tpe']:
    for num_traj in [1, 2, 5, 10, 20, 50, 100]:
        os.rename('%s/n%d_t1.dat' % (type, num_traj), '%s/n%d_t500.dat' % (type, num_traj))
