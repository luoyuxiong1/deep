import scipy.io as sio

yFile = './2013_IEEE_GRSS_DF_Contest_CASI_349_1905_144.mat'    #相对路径
datay=sio.loadmat(yFile)

print(datay)