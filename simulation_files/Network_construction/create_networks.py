import numpy as np
### Create a regular network where each connections is rewired randomly with probability p
### Save connectivity matrix and connections list in separate files
################################################# 
N = 1000
n = 10
################ create a regular network whit probability p of randon connections
def create_network(name, p):
    ### load regular network on matrix
    matrix = np.zeros((N, N))
    for i in range(N):
        for j in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]: # ten regular connections
            matrix[i, (N+i+j)%N] =1 ## add regular connection

    ### rewire whit probability p
    changed = 0 #counter for number of rewired connections
    for i in range(N):
        for j in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]: # ten regular connections
            if (np.random.uniform() < p): ## random choice, less that p then rewired connection
                matrix[i, (N+i+j)%N] = 0 ## remove regular connection
                changed +=1 
                asigned = 0 ## make sure connection is rewired
                while (asigned == 0):
                    conn = np.random.randint(1, high = N+1, size = 2) #random couple
                    if ((conn[0] != conn[1]) and (matrix[conn[0]-1, conn[1]-1] != 1)):#check it's not self connection or already connected
                        matrix[conn[0]-1, conn[1]-1] = 1 # add connection to matrix
                        asigned = 1     
    ### create list of connections
    nets = []
    for i in range(N):
        for j in range(N):
            if (matrix[i, j] ==1):
                nets.append([i+1, j+1])
    ## save network
    np.savetxt('../network_%s.dat'%name, np.array(nets), fmt = '%d')
    np.savetxt('../network_matrix_%s.dat'%name, matrix, fmt = '%d')
    print(name, p, changed, changed/(n*N))
    return 0
################################################## 
################################################# 
create_network('small', 0.01)
create_network('regular', 0.0)
create_network('random', 1.0)

################ Disconnected network: A file whit the same format full of zeros
np.savetxt('../network_disconnected.dat', np.zeros((n*N, 2)), fmt = '%d')
