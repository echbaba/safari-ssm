import numpy as np
import scipy 
import matplotlib.pyplot as plt

# This is the vanilla sequential update. 
def updateC(SSMobj,c,k,fk,alpha=0.5):
# input:    Current value of vector c
#           Matrices A and B
#           new input value fk
#           size of window, k (=current index if scaled, =W if translated) 
#           alpha, weight of bilinear transform (0 = forward, 1 = backward)
#              Default to 0.5 (bilinear)
# output:   updated coefficient vector, cnew

    A = SSMobj.A
    B = SSMobj.B

    N = A.shape[0]
    # separate this into 3 terms for readability:
    term1 = np.linalg.pinv(np.eye(N) + (1/(k))*alpha*A)
    term2 = np.eye(N) - (1/(k))*(1-alpha)*A
    term3 = (1/(k))*term1
    cnew = (term1 @ term2 @ c) + (term3 @ B * fk)
    return np.squeeze(cnew)

def compute_ssm_state(SSMobj, x,  W=1, alpha=0.5, eval_ind=[-1]):
# input:    SSM object containing A, B, and measure type
#           x, input signal    
#           W, size of window (if applicable)
#           alpha, weight of bilinear transform (0 = forward, 1 = backward, default is 0.5 for GBT)
#           eval_ind, indices of time steps to return (default is only final state)

    A = SSMobj.A
    B = SSMobj.B
    L=len(x)
    
    if SSMobj.is_diag:
        N = SSMobj.B_diag.shape[0]
        K = np.zeros([N,L]) * (0+0j) #When we diagonalize SSMs, the SSM becomes complex. until we multiply ssmobj.eig_vec and make it real again.
    else:
        N = A.shape[0]
        K = np.zeros([N,L])
    
    if SSMobj.meas == 'translated':
        if SSMobj.is_diag:
            # print("A is diagonal.")
            dt = 1/W
            mat1=1/(1 + alpha*dt* SSMobj.eig_val )
            Abar = mat1 * (1-(1-alpha)*dt* SSMobj.eig_val)
            Bbar = dt * mat1 * SSMobj.B_diag
            c= np.zeros(N)* (0+0j)
            for t in range(L):
                c= Abar*c + Bbar*x[t]
                K[:,t] = c.squeeze()
            K= K[:,eval_ind]
            K= np.real( SSMobj.eig_vec @ K )  # It's real anyways, but we convert it so the small imaginary values won't carry over
                             
        else:
            # print("A is not diagonal.")
            dt = 1/W
            mat1=np.linalg.inv(np.eye(N) + alpha*dt*A )
            Abar = mat1 @ (np.eye(N)-(1-alpha)*dt*A)
            Bbar = dt * mat1 @ B
            c= np.zeros((N,1))
            for t in range(L):
                c= Abar@c + Bbar*x[t]
                K[:,t] = c.squeeze()
            K= K[:,eval_ind]

    elif SSMobj.meas == 'scaled':
        #if SSMobj.fname=='legendre':
            #K=Fast_LegS_Solver( SSMobj, x , alpha, eval_ind )
        if SSMobj.is_diag :
            print("A is diagonal.")
            c= np.zeros(N)* (0+0j)
            for t in range(0,L):
                dt = 1/(t+1)
                mat1= 1/(  1 + alpha*dt* SSMobj.eig_val )
                Abar = mat1 * (1-(1-alpha)*dt*SSMobj.eig_val)
                Bbar = dt * mat1 * SSMobj.B_diag
                c= Abar*c + Bbar*x[t]
                K[:,t] = c.squeeze()
            K= K[:,eval_ind]
            K= np.real( SSMobj.eig_vec @ K )  #It's real anyways, but we convert it so the small imaginary values won't carry over
            
        else:
            print("A is not diagonal.")
            c= np.zeros((N,1))
            for t in range(0,L):
                dt = 1/(t+1)
                mat1=np.linalg.inv(np.eye(N) + alpha*dt*A )
                Abar = mat1 @ (np.eye(N)-(1-alpha)*dt*A)
                Bbar = dt * mat1 @ B
                c = Abar @ c +  Bbar*x[t]
                K[:,t] = c.squeeze()
            K= K[:,eval_ind]
            
    return K
        
def Fast_LegS_Solver(SSMobj,x, alpha=0.5, eval_ind=[-1]):

    N=SSMobj.A.shape[0]
    L= len(x)
    B=np.squeeze(SSMobj.B)
    D1 = np.sqrt(  2*np.arange(N)  +1   )
    D0= - np.arange(N) / ( 2* np.arange(N) +1 )
    
    Representatnios= np.zeros((N, L)) * (0+0j)
    C_GBT=np.zeros(N)
    pbar=range(0,L)
    for i in pbar:
        
        Term0= D1*C_GBT
        Term1 = np.cumsum( Term0)
        Term2= C_GBT + ( (1-alpha)/(i+1)) * np.arange(N)  * C_GBT-    ((1-alpha)/(i+1))* D1 * Term1
        y_tild = ((i+1)/ alpha ) * ( Term2 + B/(i+1)* x[i]   ) / D1
        D_tild = D0   + ((i+1)/alpha) / (D1*D1)
        
        X_tild=np.zeros(N)
        Xcumsum=0
        for q in range(N):

            X_tild[q]=  (y_tild[q]-Xcumsum)/(  1+D_tild[q] )
            Xcumsum=Xcumsum+X_tild[q]
        
        C_GBT=  X_tild/D1
        
        
        
        Representatnios[:,i]= C_GBT

    return np.real( Representatnios[:,eval_ind] )


# ### This is suboptimal. I'll be working on this in the next commits.
# def constructKernel(SSMobj, L, W=1, alpha=0.5):
# # input:    SSM object containing A, B, and measure type
# #           L, length of kernel
# #           W, size of window (if applicable)

#     A = SSMobj.A
#     B = SSMobj.B
    
#     if SSMobj.is_diag:
#         N = SSMobj.B_diag.shape[0]
#         K = np.zeros([N,L])* (0+0j)
#     else:
#         N = A.shape[0]
#         K = np.zeros([N,L])

#     if SSMobj.meas == 'translated':
#         dt = 1/W
#         if SSMobj.is_diag:
#             print("A is diagonal.")
#             mat1=1/(1 + alpha*dt*SSMobj.eig_val )
#             Abar = mat1 * (1-(1-alpha)*dt*SSMobj.eig_val)
#             Bbar = dt * mat1 * SSMobj.B_diag
#             ktmp=Bbar
#             for t in range(L):
#                 ktmp = Abar  * ktmp
#                 K[:,t] = ktmp.squeeze() 
#             K= np.real( SSMobj.eig_vec @ K )      
#         else:
#             print("A is not diagonal.")
#             mat1=np.linalg.inv(np.eye(N) + alpha*dt*A )
#             Abar = mat1 @ (np.eye(N)-(1-alpha)*dt*A)
#             Bbar = dt * mat1 @ B
#             ktmp=Bbar
#             for t in range(L):
#                 K[:,t] = ktmp.squeeze()
#                 ktmp = Abar @ ktmp 

#     elif SSMobj.meas == 'scaled':
#         if SSMobj.is_diag :
#             print("A is diagonal.")
#             Atmp = np.ones(N)* (1+0j)
#             for t in range(0,L):
#                 dt = 1/(L-t)
#                 mat1= 1/(  1 + alpha*dt* SSMobj.eig_val )
#                 Abar = mat1 * (1-(1-alpha)*dt*SSMobj.eig_val)
#                 Bbar = dt * mat1 * SSMobj.B_diag
#                 ktmp = Atmp @ Bbar
#                 Atmp = Atmp @ Abar
#                 K[:,t] = ktmp.squeeze()
#             K= np.real( SSMobj.eig_vec @ K )  #It's real anyways, but we convert it so the small imaginary values won't carry over
#         else:
#             print("A is not diagonal.")
#             Atmp = np.eye(N)
#             for t in range(0,L):
#                 dt = 1/(L-t)
#                 mat1=np.linalg.inv(np.eye(N) + alpha*dt*A )
#                 Abar = mat1 @ (np.eye(N)-(1-alpha)*dt*A)
#                 Bbar = dt * mat1 @ B
#                 ktmp = Atmp @ Bbar
#                 Atmp = Atmp @ Abar
#                 K[:,t] = ktmp.squeeze()
            
#     return K



def reconstruct(c,k,D):
# reconstruct signal up to kth point based on input thus far
# input:    c, coefficient vector          
#           W, size of window
#           L, length of basis
#           D, dual of frame
# output:   g, reconstructed signal

    L = D.shape[1]

    g = D.T @ c*L
    g= np.squeeze(g)
    gscaled = scipy.signal.resample(g,k)
    return gscaled


def resample(x, L2):
    L=len(x)
    if np.iscomplexobj(x):
        Z= np.zeros(3*L) * (1j)
    else:
        Z= np.zeros(3*L)
    Z[0:L]=np.flip(x)
    Z[L:2*L]=x
    Z[2*L:3*L]=np.flip(x)
    Y= scipy.signal.resample(Z, 3*L2)
    return Y[L2:2*L2]
    


