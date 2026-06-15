import numpy as np

from flow.read_utils import read_par_bin
# from flow.evolution import Evolution

s = 0.002
eta = np.array([0.1,0.1234567])
g=15.5
# par_file = open('test_flow_parameters.bin', 'wb+')
# def write_files_params():
#     param = np.concatenate([[s], eta, [g]])
#     np.savetxt(par_file, param, delimiter=' ', newline=' ')
#     param.tofile(par_file)
#
#     write_files_params()
#     write_files_params()
#     write_files_params()
# par_file.close()

r=read_par_bin('', 'test_flow_parameters.bin', 4, type=np.float64)
print(r)