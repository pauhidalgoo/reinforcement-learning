import numpy as np
import matplotlib.pyplot as plt


def print_policy(policy_matrix2, V=np.array([]),lstates=[]):
    """
    The function `print_policy` takes a policy matrix, a value matrix, and a list of states as input,
    and visualizes the policy and value matrices using matplotlib.
    
    :param policy_matrix2: The policy matrix is a 2D array that represents the policy for each state in
    a grid world. Each element in the matrix corresponds to a state, and the value represents the action
    to take in that state. The possible actions are represented by integers: -1 for no action, 0 for
    :param V: The V parameter is a numpy array that represents the values of each state in the grid. It
    is used to color the grid cells based on their values. If V is not provided, the grid cells will be
    colored uniformly
    :param lstates: The parameter "lstates" is a list of states. Each state is represented by a tuple
    containing the row and column indices of the state in the policy matrix. These states will be
    highlighted in the visualization by setting their corresponding values in the V matrix to 0.7
    """
    policy_matrix = policy_matrix2[::-1]
    shape = policy_matrix.shape
    U=np.zeros(policy_matrix.shape)
    R=np.zeros(policy_matrix.shape)
    for row in range(shape[0]):
        for col in range(shape[1]):
            if(policy_matrix[row,col] == -1): 
                U[row,col]=0    
                R[row,col]=0  
            elif(policy_matrix[row,col] == 0): 
                U[row,col]=0    
                R[row,col]=1 
                #policy_string += " ^  "
            elif(policy_matrix[row,col] == 1): 
                U[row,col]=1   
                R[row,col]=0 
                #policy_string += " >  "
            elif(policy_matrix[row,col] == 2): 
                U[row,col]=0    
                R[row,col]=-1 
                #policy_string += " v  "           
            elif(policy_matrix[row,col] == 3):
                U[row,col]=-1   
                R[row,col]=0                 
                #policy_string += " <  "
            elif(np.isnan(policy_matrix[row,col])): 
                U[row,col]=0    
                R[row,col]=0  
                #policy_string += " #  "
    plt.rcParams['figure.figsize'] = (4,3)
    if V.size==0:
        V=np.ones(policy_matrix.shape)
        for x in lstates:
            V[x.multi_index]=0.7
        plt.matshow(V,extent=[0, 4, 0, 3],vmin=0, vmax=1)
    else:
        plt.matshow(V,extent=[0, 4, 0, 3])
    plt.grid()
    X, Y = np.meshgrid(np.arange(0.5, 4.5, 1), np.arange(0.5, 3.5, 1))
    Q = plt.quiver(X, Y,U,R)

    plt.show(block=False)


