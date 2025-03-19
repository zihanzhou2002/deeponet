import numpy as np
import torch

def model(X, net): # take in num_datax1
    x_func=net.branch(torch.tensor(X[0])).to(torch.float32) # output num_datax 2x2, num_datax 4
    # num_datax4
    #b=torch.reshape(b, (2, 2)) #num_datax 2x2
    #print(f"x_fun = {x_func}")
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1]))).to(torch.float32)
    #print(f"x_loc = {x_loc}")
    
    # Split x_func into respective outputs
    shift = 0
    size = x_loc.shape[1]
    xs = []
    for i in range(net.num_outputs):
        x_func_ = x_func[:, shift : shift + size]
        x = net.merge_branch_trunk(x_func_, x_loc, i)
        xs.append(x)
        shift += size
    
    result = net.concatenate_outputs(xs)
    #print(f"result = {result}")
    return result.to(torch.float32)

def model_matrix_batch(X, net): # take in num_datax1
    x_func=net.branch(torch.tensor(X[0])) # output num_datax 2x2, num_datax 4
    
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1])))

    x_func = x_func.view(x_func.shape[0], 2, -1).to(torch.float64)
    x_loc = x_loc.unsqueeze(2).to(torch.float64)
    
    #b =  torch.tensor([net.b[0], net.b[1]], dtype=torch.float64).unsqueeze(1)
    #b_batch = b.expand(x_func.shape[0], -1, -1).to(torch.float64)
    
    result = torch.bmm(x_func, x_loc).to(torch.float64) #+ b_batch
    #print(f"result: {result.shape}")
    return result.squeeze()

def model_energy_v2(X,net, omega): # take in num_datax1
    x_func=net.branch(torch.tensor(X[0])) # output num_datax 2x2, num_datax 4
    
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1])))
    W = torch.tensor([[omega**2, 0], [0, 1]]).double()
    W_sqrt = torch.tensor([[omega, 0], [0, 1]]).double()
    # Split x_func into respective outputs

    xs = []
    for i in range(len(x_func)):
        p0, q0 = X[0][i]
        x_func_ = torch.reshape(x_func[i],(2, -1) ).double()
        x_loc_ = x_loc[i].double()
        
        E = 0.5* (omega*p0**2 +q0**2)
        B_tilde = torch.mm(W_sqrt, x_func_).double()
        Q_tilde, R = torch.linalg.qr(B_tilde)
        alpha_tilde = torch.mm(R, x_loc_.unsqueeze(1)).double()
        
        Q_tilde = Q_tilde.to(torch.float64)
        R = R.to(torch.float64)
        alpha_scaled = alpha_tilde* np.sqrt(E) / torch.linalg.vector_norm(alpha_tilde)
        W_sqrt_inv = torch.linalg.inv(W_sqrt)
        
        W_sqrt_inv = W_sqrt_inv.to(torch.float64)
        temp = torch.mm(W_sqrt_inv, Q_tilde).double()
        x = torch.mm(temp, alpha_scaled).double() + torch.tensor([net.b[0], net.b[1]], dtype = torch.float64).unsqueeze(1)
        xs.append(x.squeeze())

    result = torch.stack(xs, dim=0).to(torch.float64)

    return result

def model_energy_v2_batch(X,net, omega): # take in num_datax1
    x_func=net.branch(torch.tensor(X[0])) # output num_datax 2x2, num_datax 4
    
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1])))
    # Split x_func into respective outputs
    
    x_func = x_func.view(x_func.shape[0], 2, -1).to(torch.float64)
    x_loc = x_loc.unsqueeze(2).to(torch.float64)

    p0 = torch.tensor(X[0][:, 0])
    q0 = torch.tensor(X[0][:, 1])
    
    E = 0.5* ((omega*p0)**2 +q0**2)
    
    W_sqrt = torch.tensor([[omega, 0], [0, 1]]).double()
    W_sqrt_inv = torch.linalg.inv(W_sqrt)
    W_sqrt_batch = W_sqrt.expand(x_func.shape[0], -1, -1)
    W_sqrt_inv_batch = W_sqrt_inv.expand(x_func.shape[0], -1, -1)
    
    B_tilde = torch.bmm(W_sqrt_batch, x_func).double()
    
    Q_tilde, R = torch.linalg.qr(B_tilde)
    
    alpha_tilde = torch.bmm(R, x_loc).double()
    
    b = torch.tensor([net.b[0], net.b[1]], dtype = torch.float64).unsqueeze(1)
    b_batch = b.expand(x_func.shape[0], -1, -1)
    
    norm_alpha_tilde = torch.linalg.vector_norm(alpha_tilde, dim=1, keepdim=True)
    alpha_scaled = alpha_tilde* torch.sqrt(2*E).unsqueeze(1).unsqueeze(2) / norm_alpha_tilde
    
    basis = torch.bmm(W_sqrt_inv_batch, Q_tilde).double()
    result = torch.bmm(basis, alpha_scaled) #+ b_batch
    
    return result.squeeze()

def model_energy(X,net,omega): # take in num_datax1
    x_func=net.branch(torch.tensor(X[0])) # output num_datax 2x2, num_datax 4
    
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1])))
    # Split x_func into respective outputs

    xs = []
    for i in range(len(x_func)):
        p0, q0 = X[0][i]
        x_func_ = torch.reshape(x_func[i],(2, -1) ).double()
        x_loc_ = x_loc[i].double()
        
        E = 0.5* (omega*p0**2 +q0**2)
        
        Q, R = torch.linalg.qr(x_func_)
        alpha_tilde = torch.mm(R, x_loc_.unsqueeze(1)).double()
        
        Q = Q.to(torch.float64)
        R = R.to(torch.float64)
        alpha_scaled = alpha_tilde* np.sqrt(E) / torch.linalg.vector_norm(alpha_tilde)
        
        b = torch.tensor([net.b[0], net.b[1]], dtype = torch.float64).unsqueeze(1)

        x = torch.mm(Q, alpha_scaled).double() + b
        xs.append(x.squeeze())

    result = torch.stack(xs, dim=0).to(torch.float64)
    return result

def model_energy_batch(X,net, omega): # take in num_datax1
    x_func=net.branch(torch.tensor(X[0])) # output num_datax 2x2, num_datax 4
    
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1])))
    # Split x_func into respective outputs
    
    x_func = x_func.view(x_func.shape[0], 2, -1).to(torch.float64)
    x_loc = x_loc.unsqueeze(2).to(torch.float64)

    p0 = torch.tensor(X[0][:, 0])
    q0 = torch.tensor(X[0][:, 1])
    
    E = 0.5* (omega*p0**2 +q0**2)

    Q, R = torch.linalg.qr(x_func)
    
    alpha_tilde = torch.bmm(R, x_loc).double()
    
    b = torch.tensor([net.b[0], net.b[1]], dtype = torch.float64).unsqueeze(1)
    b_batch = b.expand(x_func.shape[0], -1, -1)
    
    norm_alpha_tilde = torch.linalg.vector_norm(alpha_tilde, dim=1, keepdim=True)
    alpha_scaled = alpha_tilde* torch.sqrt(2*E).unsqueeze(1).unsqueeze(2) / norm_alpha_tilde
    
    result_energy = torch.bmm(Q, alpha_scaled) + b_batch
    result_energy = result_energy.double()
    result_orig = torch.bmm(x_func, x_loc) + b_batch
    result_orig = result_orig.double()
    return result_energy.squeeze()

def model_energy_v3(X,net, omega): # take in num_datax1
    x_func=net.branch(torch.tensor(X[0])) # output num_datax 2x2, num_datax 4
    
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1])))
    # Split x_func into respective outputs

    xs = []
    for i in range(len(x_func)):
        p0, q0 = X[0][i]
        x_func_ = torch.reshape(x_func[i],(2, -1) ).double()
        x_loc_ = x_loc[i].double()
        
        E = 0.5* (omega*p0**2 +q0**2)
        
        Q, R = torch.linalg.qr(x_func_)
        alpha1 = torch.mm(R, x_loc_.unsqueeze(1)).double()
        
        Q = Q.to(torch.float64)
        R = R.to(torch.float64)
        
        b = torch.tensor([net.b[0], net.b[1]], dtype = torch.float64).unsqueeze(1)
        alpha2 = torch.linalg.solve(Q, b)
        alpha2 = alpha2.to(torch.float64)
        
        alpha_tilde = alpha1 + alpha2
        
        alpha_scaled = alpha_tilde* np.sqrt(E) / torch.linalg.vector_norm(alpha_tilde)
        x = torch.mm(Q, alpha_scaled).double() 
        xs.append(x.squeeze())

    #print(f"xs {len(xs), len(xs[0])}")
    #result = net.concatenate_outputs(xs)
    #print(f"result = {result}")
    result = torch.stack(xs, dim=0).to(torch.float64)
    #print(f"result: {result.shape}")
    return result

def model_energy_v3_batch(X,net, omega): # take in num_datax1
    x_func=net.branch(torch.tensor(X[0])) # output num_datax 2x2, num_datax 4
    
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1])))
    # Split x_func into respective outputs
    
    x_func = x_func.view(x_func.shape[0], 2, -1).to(torch.float64)
    x_loc = x_loc.unsqueeze(2).to(torch.float64)

    p0 = torch.tensor(X[0][:, 0])
    q0 = torch.tensor(X[0][:, 1])
    
    E = 0.5* (omega*p0**2 +q0**2)

    Q, R = torch.linalg.qr(x_func)
    
    alpha1 = torch.bmm(R, x_loc).double()
    
    b = torch.tensor([net.b[0], net.b[1]], dtype = torch.float64).unsqueeze(1)
    b_batch = b.expand(x_func.shape[0], -1, -1)
    
    alpha2 = torch.linalg.solve(Q, b_batch).double()
    
    alpha_tilde = alpha1 + alpha2
    norm_alpha_tilde = torch.linalg.vector_norm(alpha_tilde, dim=1, keepdim=True)
    alpha_scaled = alpha_tilde* torch.sqrt(E).unsqueeze(1).unsqueeze(2) / norm_alpha_tilde
    
    result = torch.bmm(Q, alpha_scaled).double()
    return result.squeeze(2)
    

def model_matrix(X,net,omega): # take in num_datax1
    x_func=net.branch(torch.tensor(X[0])) # output num_datax 2x2, num_datax 4
    
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1])))
    W = torch.tensor([[omega, 0], [0, 1]])
    W_sqrt = torch.tensor([[np.sqrt(omega), 0], [0, 1]])
    # Split x_func into respective outputs

    xs = []
    for i in range(len(x_func)):
        x_func_ = torch.reshape(x_func[i],(2, -1) )
        x_loc_ = x_loc[i]
        x = torch.mm(x_func_, x_loc_.unsqueeze(1)) + torch.tensor([net.b[0], net.b[1]]).unsqueeze(1)
        xs.append(x.squeeze())

    #print(f"xs {len(xs), len(xs[0])}")
    #result = net.concatenate_outputs(xs)
    #print(f"result = {result}")
    result = torch.stack(xs, dim=0)
    #print(f"result: {result.shape}")
    return result

