# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D

##################################
########## init ##################
##################################

def main():
    ##### parse the inputs ###########
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action="store", dest="data_file",
        default="./data/Yumi/S2/",
        help="Data file or Directory")
    parser.add_argument('-p', action="store", dest="param_file",
        default="./data/Yumi/task_params.csv",
        help="Data file or Task parameters (subject, tilt, starting \
                side, pegs picked)")
    parser.add_argument('-r', '--recursive',
        help="Do recursive exploring of a directory")
    args = parser.parse_args()


    s1 = data_x[:82,7:10]
    #x = list(set(s1[:,0]))
    #y = list(set(s1[:,1]))
    #z = list(set(s1[:,2]))
    #
    #

    #x = s1[:,0]
    #y = s1[:,1]
    #z = s1[:,2]
    ##y = y[:64]
    ##z = z[:64]

    _,x_index = np.unique(s1[:,0],return_index = True)

    _,y_index = np.unique(s1[:,1],return_index = True)

    _,z_index = np.unique(s1[:,2],return_index = True)

    xyz_index = np.concatenate((x_index,y_index,z_index),axis = None)

    xyz_index = np.sort(np.unique(xyz_index))

    s1 = data_x[xyz_index,7:10]

    x = s1[:,0]
    y = s1[:,1]
    z = s1[:,2]




    #print(x)
    ##
    tck, u = interpolate.splprep([x,y,z],s = 2)
    #print(tck)
    u_fine = np.linspace(0,1,30)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
    x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)

    fig2 = plt.figure(2)
    ax3d = fig2.add_subplot(111, projection='3d')
    ax3d.plot(x, y, z, 'b')
    ax3d.plot(x_knots, y_knots, z_knots, 'go')
    ax3d.plot(x_fine, y_fine, z_fine, 'g')
    fig2.show()
    plt.show()

    #
    #
    #
    #
    #
    ##
    # 3D example
    #total_rad = 10
    #z_factor = 3
    #noise = 0.1
    #
    #num_true_pts = 200
    #s_true = np.linspace(0, total_rad, num_true_pts)
    #x_true = np.cos(s_true)
    #y_true = np.sin(s_true)
    #z_true = s_true/z_factor
    #
    #num_sample_pts = 80
    #s_sample = np.linspace(0, total_rad, num_sample_pts)
    #x_sample = np.cos(s_sample) + noise * np.random.randn(num_sample_pts)
    #y_sample = np.sin(s_sample) + noise * np.random.randn(num_sample_pts)
    #z_sample = s_sample/z_factor + noise * np.random.randn(num_sample_pts)
    #
    #tck, u = interpolate.splprep([x_sample,y_sample,z_sample], s=2)
    #x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
    #u_fine = np.linspace(0,1,num_true_pts)
    #x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
    #
    #fig2 = plt.figure(2)
    #ax3d = fig2.add_subplot(111, projection='3d')
    #ax3d.plot(x_true, y_true, z_true, 'b')
    #ax3d.plot(x_sample, y_sample, z_sample, 'r*')
    #ax3d.plot(x_knots, y_knots, z_knots, 'go')
    #ax3d.plot(x_fine, y_fine, z_fine, 'g')
    #fig2.show()
    #plt.show()

if __name__ == '__main__':
    main()
