import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import pickle
import time
import matplotlib.pyplot as plt
from typing import Sequence, Generic, Callable
import optax
from functools import partial
from jax import grad, jit, vmap, jacfwd



class LearnedMetric(nn.Module):
    """
    Parameters
    ----------
        dim         : Dimension of manifold/input.
        n_units     : Number of units in each layer.
        activation  : Nonlinearity between each layer.
    """
    dim: int
    n_units: Sequence[int] = (16, 32, 16)
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.softplus

    def setup(self):
        self.n_hidden = len(self.n_units)
        self.layers = [nn.Dense(f) for f in self.n_units]

    @nn.compact
    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != self.n_hidden - 1:
                x = self.activation(x)

        #L_x = nn.Dense(self.dim * (self.dim - 1) // 2, name='L')(x)
        D_x = nn.Dense(self.dim+1, name='D')(x)
        D_x = nn.softplus(D_x)

        # jnp.diag(D_x) for matrix D
        # utils.fill_lower_tri(L_x, self.dim) for matrix L
        return D_x


def create_train_state(rng, model, optimizer):
    rng, init_rng = jax.random.split(rng)
    params = model.init(rng, jnp.ones([1, 3]))['params']
    opt_state = optimizer.init(params)
    return params, opt_state, init_rng

def T2_simple_collocation(key , n ):
    """
    Samples n points for phi and theta over [0,2pi] and time over [0,1]
    Parameters
    ----------
    key - randomiser key
    n   - number of points

    Returns
    -------
    stacked array of time , theta, phi
    """
    key, key_phi, key_theta , key_time = jax.random.split(key, 4)
    phi = jax.random.uniform(key_phi, shape=(n,)) * 2 * jnp.pi
    theta = jax.random.uniform(key_theta, shape=(n,)) * 2 * jnp.pi
    time = jax.random.uniform(key_time, shape=(n,))*2
    return jnp.stack([time, theta, phi], axis=-1)

def T2_simple_initial(key , n):
    key, key_phi, key_theta , key_time = jax.random.split(key, 4)
    time = jax.random.uniform(key_time, shape=(n,))*0
    phi = jax.random.uniform(key_phi, shape=(n,)) * 2 * jnp.pi
    theta = jax.random.uniform(key_theta, shape=(n,)) * 2 * jnp.pi
    return jnp.stack([time, theta, phi], axis=-1)

def cigar_simple_initial(key , n):
    key, key_phi, key_theta , key_time = jax.random.split(key, 4)
    time = jax.random.uniform(key_time, shape=(n,))*0
    phi = jax.random.uniform(key_phi, shape=(n,)) * 10
    theta = jax.random.uniform(key_theta, shape=(n,)) * 10
    return jnp.stack([time, theta, phi], axis=-1)

def T2_g_0_generator(n , Data_i):
  theta = Data_i[:,1]
  phi = Data_i[:,2]

  g11 = (2+jnp.cos(phi))**2
  g12 = jnp.zeros((n,))
  g21 = jnp.zeros((n,))
  g22 = jnp.ones((n,))
  return jnp.stack([g11, g12, g21, g22], axis=-1).reshape(n, 2, 2)

def cigar_g_0_generator(n, Data_i):
    theta = Data_i[:, 1]
    phi = Data_i[:, 2]
    g11 = (1 + phi**2+theta**2) ** -1
    g12 = jnp.zeros((n,))
    g21 = jnp.zeros((n,))
    g22 = (1 + phi**2+theta**2) ** -1
    return jnp.stack([g11, g12, g21,g22], axis=-1).reshape(n,2,2)

def metric_matrix(p, params):
    return LearnedMetric(p.shape[-1], (16, 32, 16)).apply({'params': params}, p) #.reshape(32,2,2)

def get_metric(p, params , metric_out):
    dim = p.shape[-1]
    D = metric_out(p,params)

    g = D.reshape(2,2)
    gT = jnp.transpose(g,(1,0))
    return jnp.matmul(gT,g)

def inverse_metric_matrix(p, params, metric_out):
    g = get_metric(p, params, metric_out)
    return jnp.linalg.inv(g)

def dgdt_finder(p,params,metric_out):
    derivatives = jacfwd(get_metric, argnums=0)(p, params, metric_out)  # [..., i,j,l]
    g_t = derivatives[:,:,0]
    return g_t

def christoffel_symbols(p, params, metric_out):
    g_inv = inverse_metric_matrix(p, params, metric_out)
    derivatives = jacfwd(get_metric, argnums=0)(p, params, metric_out)  # [..., i,j,l]
    g_t = derivatives[:,:,0]
    jac_g = derivatives[:,:,1:]

    x1 = jnp.einsum('...kl, ...jli->...kij', g_inv, jac_g)
    x2 = jnp.einsum('...kl, ...ilj->...kij', g_inv, jac_g)
    x3 = jnp.einsum('...kl, ...ijl->...kij', g_inv, jac_g)

    cs = 0.5 * (x1 + x2 - x3)
    return cs


def riemann_curvature_tensor(p, params, metric_out):
    # [..., i,j,k,l]
    derivatives_cs = jacfwd(christoffel_symbols, argnums=0)(p, params, metric_out)
    jac_cs = derivatives_cs[:, :, :, 1:]
    cs = christoffel_symbols(p, params, metric_out)

    x1 = jnp.einsum('...iljk->...ijkl', jac_cs)
    x2 = jnp.einsum('...ikjl->...ijkl', jac_cs)
    x3 = jnp.einsum('...ikm, ...mlj->...ijkl', cs, cs)
    x4 = jnp.einsum('...ilm, ...mkj->...ijkl', cs, cs)

    riemann = x1 - x2 + x3 - x4
    return riemann


def ricci_tensor(p, params, metric_out):
    riemann = riemann_curvature_tensor(p, params, metric_out)
    ricci = jnp.einsum('...kikj->...ij', riemann)
    # print(ricci.shape)
    return ricci

def ricci_scalar(p, params, metric_out):

    g_inv = inverse_metric_matrix(p, params, metric_out)
    ricci_t = ricci_tensor(p, params, metric_out)
    R = jnp.einsum('...ij, ...ij->...', g_inv, ricci_t)
    return R

def grad_ricci_scalar(p,params,metric_out):
  grad_R = jacfwd(ricci_scalar,argnums=0)(p,params,metric_out)
  #print(grad_R.shape)
  return grad_R

def residual_loss(p, params,metric_out,key):
  key1, key2 = jax.random.split(key)
  dgdt = vmap(dgdt_finder, in_axes=(0,None,None))(p, params, metric_out)
  ricci = vmap(ricci_tensor , in_axes=(0,None,None))(p,params,metric_out)
  time = p[:,0]
  g = vmap(get_metric, in_axes=(0, None, None))(p, params, metric_out)
  avg_r = vmap(average_curvature , in_axes=(0,None,None,None))(time,params,metric_out,key)
  avg_r = avg_r.reshape(time.shape[0],1,1)
  a = jnp.square(dgdt + 2*ricci - avg_r*g)
  b = jnp.mean(a)
  return b
#@jit
def initial_loss(n, params, metric_out, Data_i):
  #key = jax.random.PRNGKey(2)
  #Data_i = T2_simple_initial(key , n)
  g = vmap(get_metric, in_axes=(0,None,None))(Data_i, params, metric_out)

  g_0 = T2_g_0_generator(n,Data_i)
  a = jnp.square(g-g_0)
  b = jnp.mean(a)
  return 10*b

def periodic_loss(n,params,metric_out,Data_r):

    g = vmap(get_metric, in_axes=(0,None,None))(Data_r, params, metric_out)
    Data_r_transformed = periodic_transformer(Data_r)
    g_transformed = vmap(get_metric, in_axes=(0,None,None))(Data_r_transformed, params, metric_out)
    a = jnp.square(g - g_transformed)
    b = jnp.mean(a)
    return b

def periodic_transformer(Data):
    time = Data[:,0]
    theta = Data[:,1]
    phi = Data[:,2]
    theta_transformed = 2*jnp.pi - theta
    phi_transformed = 2*jnp.pi - phi
    return jnp.stack([time, theta_transformed, phi_transformed], axis=-1)

def loss(Data_r,Data_i,params,metric_out,n,key):
  loss_res = residual_loss(Data_r,params,metric_out,key)
  loss_init = initial_loss(n,params,metric_out,Data_i)
  #loss_periodic = periodic_loss(n,params , metric_out , Data_r)
  loss_sym = sym_loss(n,params , metric_out , Data_r,key)
  loss_ends = end_loss(n , params , metric_out , Data_r)
  loss_R_grad = ricci_grad_end_loss(params,metric_out,Data_r)
  loss_R_init = ricci_initial_loss(params,metric_out,Data_r)
  l = loss_init + 3*loss_res + 10*loss_sym +4*loss_ends + 10*loss_R_grad + 10*loss_R_init
  #l = loss_init + loss_res + loss_periodic
  #print(jnp.array([l]))
  return l

def sym_loss(n, params, metric_out, Data_r, key):
    g = vmap(get_metric, in_axes=(0, None, None))(Data_r, params, metric_out)
    Data_r_transformed_theta = sym_transformer(Data_r, 0, key)
    Data_r_transformed_phi = sym_transformer(Data_r, 1, key)
    g_transformed_theta = vmap(get_metric, in_axes=(0, None, None))(Data_r_transformed_theta, params, metric_out)
    g_transformed_phi = vmap(get_metric, in_axes=(0, None, None))(Data_r_transformed_phi, params, metric_out)
    a = jnp.square(g - g_transformed_theta)
    b = jnp.square(g - g_transformed_phi)
    c = jnp.mean(a) + jnp.mean(b)
    return c


def grad_periodic_loss(n, params, metric_out, Data_r):
    grad_g = vmap(gradient_tensor, in_axes=(0, None, None))(Data_r, params, metric_out)
    Data_r_transformed = periodic_transformer(Data_r)
    grad_g_transformed = vmap(gradient_tensor, in_axes=(0, None, None))(Data_r_transformed, params, metric_out)
    differences = jnp.square(grad_g - grad_g_transformed)
    b = jnp.mean(differences)
    return b

def sym_transformer(Data, variable, key):
    time = Data[:, 0]
    theta = Data[:, 1]
    phi = Data[:, 2]
    n = phi.shape[-1]
    if variable == 0:
        theta = jax.random.uniform(key, shape=(n,)) * 2 * jnp.pi
    else:
        phi = 2 * jnp.pi - phi
    return jnp.stack([time, theta, phi], axis=-1)


def end_loss(n, params, metric_out, Data_r):
    time = Data_r[:, 0]
    theta = Data_r[:, 1]
    phi_0 = Data_r[:, 2] * 0.0
    phi_2pi = ((Data_r[:, 2] * 0.0) + 1) * 2 * jnp.pi
    data_0 = jnp.stack([time, theta, phi_0], axis=-1)
    data_2pi = jnp.stack([time, theta, phi_2pi], axis=-1)

    g_0 = vmap(get_metric, in_axes=(0, None, None))(data_0, params, metric_out)
    g_2pi = vmap(get_metric, in_axes=(0, None, None))(data_2pi, params, metric_out)
    return jnp.mean(jnp.square(g_0 - g_2pi))

def end_transformer(p, variable, value):
    time = p[:, 0]
    u = p[:, 1]
    v = p[:, 2]
    if variable == 0:
        u = (u * 0 + 1) * value
        return jnp.stack([time, u, v], axis=-1)
    else:
        v = (v * 0 + 1) * value
        return jnp.stack([time, u, v], axis=-1)

def ricci_initial_loss(params, metric_out, Data):
    time = Data[:, 0]
    u = Data[:, 1]
    v = Data[:, 2]
    time = time * 0
    data = jnp.stack([time, u, v], axis=-1)
    R = vmap(ricci_scalar, in_axes=(0, None, None))(data, params, metric_out)
    R_0 = 2 * jnp.cos(v) / (2 + jnp.cos(v))
    R_loss = jnp.mean(jnp.square(R - R_0))
    return R_loss


def ricci_grad_end_loss(params, metric_out, Data_r):
    p_u_0 = end_transformer(Data_r, 0, 0)
    p_u_2p = end_transformer(Data_r, 0, 2 * jnp.pi)
    p_v_0 = end_transformer(Data_r, 1, 0)
    p_v_2p = end_transformer(Data_r, 1, 2 * jnp.pi)

    grad_R_u_0 = vmap(grad_ricci_scalar, in_axes=(0, None, None))(p_u_0, params, metric_out)
    grad_R_u_2p = vmap(grad_ricci_scalar, in_axes=(0, None, None))(p_u_2p, params, metric_out)
    grad_R_v_0 = vmap(grad_ricci_scalar, in_axes=(0, None, None))(p_v_0, params, metric_out)
    grad_R_v_2p = vmap(grad_ricci_scalar, in_axes=(0, None, None))(p_v_2p, params, metric_out)
    loss = 4 * (jnp.mean(jnp.square(grad_R_u_0 - grad_R_u_2p)) + jnp.mean(jnp.square(grad_R_v_0 - grad_R_v_2p)))
    return loss

def train_metric_g(params_g, opt_state_g, optimizer_g, data_r,data_i,counter , hist,key):
    #f, grad_f = vmap(jax.value_and_grad(scalar_f, argnums=0), in_axes=(0, None))(data, params_f)

    grads_g = jax.grad(loss, argnums=2)(data_r,data_i, params_g,metric_matrix,1000,key)
    param_updates_g, opt_state_g = optimizer_g.update(grads_g, opt_state_g, params_g)
    params_g = optax.apply_updates(params_g, param_updates_g)
    if counter % 20 == 0:
      l = loss(data_r,data_i, params_g,metric_matrix,1000,key)
      print(l)
      hist.append(l)
    return params_g, opt_state_g

def sqrtdetg(data, params, metric_out):
    # print('data',data.shape)
    # g = vmap(get_metric, in_axes=(0,None,None))(data, params, metric_out)
    g = get_metric(data, params, metric_out)
    # print('g shape' , g.shape)
    detg = g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0]
    # print(detg)
    return jnp.sqrt(detg)

def volume_from_time(data, params, metric_out):
    # n = x.shape[-1]
    # time_arr = jnp.ones((n,))*time
    # data = jnp.stack([time_arr,x,y],axis=-1)
    # print(data.shape)
    # dv = vmap(sqrtdetg, in_axes=(0,None,None))(data, params, metric_out)
    dv = sqrtdetg(data, params, metric_out)
    return dv

def manifold_volume(time, params, metric_out, key):
    # print(data[0])
    key_x, key_y = jax.random.split(key)
    n = 40
    time = jnp.ones((n,)) * time

    x = jax.random.uniform(key_x, shape=(n,)) * 2 * jnp.pi
    y = jax.random.uniform(key_y, shape=(n,)) * 2 * jnp.pi
    # print(time,x,y)
    data = jnp.stack([time, x, y], axis=-1)
    # print(data.shape)
    volumes = vmap(volume_from_time, in_axes=(0, None, None))(data, params, metric_out)
    # print(volumes)
    return jnp.mean(volumes)

def average_curvature_element(time, params, metric_out, key):
    key_x, key_y = jax.random.split(key)
    n = 40
    time = jnp.ones((n,)) * time

    x = jax.random.uniform(key_x, shape=(n,)) * 2 * jnp.pi
    y = jax.random.uniform(key_y, shape=(n,)) * 2 * jnp.pi
    # print(time,x,y)
    data = jnp.stack([time, x, y], axis=-1)
    # print(data.shape)
    dv = vmap(volume_from_time, in_axes=(0, None, None))(data, params, metric_out)
    R = vmap(ricci_scalar, in_axes=(0, None, None))(data, params, metric_out)
    return jnp.mean(dv * R)

def average_curvature(time, params, metric_out, key):
    avg_r = average_curvature_element(time, params, metric_out, key) / manifold_volume(time, params, metric_out, key)
    return avg_r


######################################################
######################################################

collocation_point_sampler =  partial(T2_simple_collocation, n=1000)
initial_point_sampler = partial(T2_simple_initial,n=1000)

seed = int(time.time())
rng = jax.random.PRNGKey(seed)
rng, init_rng ,vol_key= jax.random.split(rng,3)
optimizer = optax.adam(learning_rate = 3e-4)
model_g = LearnedMetric(3)
params, opt_state, init_rng = create_train_state(init_rng, model_g,  optimizer)

params = pickle.load(open('param_norm.pkl','rb'))
opt_state = pickle.load(open('opt_state_norm.pkl','rb'))

"""
vol_arr = []
curvature_arr = []
times = np.linspace(0, 2, 20)
# print('######',times)
for i in range(len(times)):
    a = (2 * jnp.pi) ** 2 * jnp.mean(manifold_volume(times[i], params, metric_matrix, vol_key))
    # print(manifold_volume(times[i],params,metric_matrix,key))
    # print(manifold_volume(times[i],params,metric_matrix,key))
    # b = jnp.mean(average_curvature(times[i], params, metric_matrix, key1)) * (2 * jnp.pi) ** 2
    vol_arr.append(a)
    #curvature_arr.append(b / a)
# print(vol_arr)
plt.plot(times, vol_arr)
plt.show()
#plt.plot(times, curvature_arr)
#plt.show()

"""

hist = []
t0 = time.time()
for t in range(int(1)):
  if t % 20 == 0:
    print('time:',time.time()-t0)
    print(f'Iteration {t} =====>>>')
  rng, sample_rng_col , sample_rng_initial,data_key = jax.random.split(rng , 4)
  Data_r = collocation_point_sampler(key = sample_rng_col)
  Data_i = initial_point_sampler(key = sample_rng_initial)
  (params, opt_state) = train_metric_g(params,opt_state, optimizer, Data_r,Data_i,t,hist,data_key)
  #if t%20 == 0:
    #pickle.dump(params, open('torus_metric.pkl','wb'))
    #pickle.dump(opt_state, open('torus_metric_norm.pkl','wb'))



######################################################
################       PLOTS     #####################
######################################################




seed = int(3)  #time.time())  # 42
key = jax.random.PRNGKey(seed)
key, key_phi, key_theta , key_time = jax.random.split(key, 4)
time = jnp.ones((1000,))*0.5
phi = jax.random.uniform(key_phi, shape=(1000,)) * 2 * jnp.pi
theta = jax.random.uniform(key_theta, shape=(1000,)) * 2 * jnp.pi
test_data5 = jnp.stack([time, theta, phi], axis=-1)

time = jnp.ones((1000,))*0.6
test_data6 = jnp.stack([time, theta, phi], axis=-1)
time = jnp.ones((1000,))*0.7
test_data7 = jnp.stack([time, theta, phi], axis=-1)
time = jnp.ones((1000,))*0.8
test_data8 = jnp.stack([time, theta, phi], axis=-1)
time = jnp.ones((1000,))*0.9
test_data9 = jnp.stack([time, theta, phi], axis=-1)
time = jnp.ones((1000,))*1
test_data1 = jnp.stack([time, theta, phi], axis=-1)
time = jnp.ones((1000,))*0
test_data0 = jnp.stack([time, theta, phi], axis=-1)
time = jnp.ones((1000,))*1.5
test_data15 = jnp.stack([time, theta, phi], axis=-1)
time = jnp.ones((1000,))*1.3
test_data13 = jnp.stack([time, theta, phi], axis=-1)
time = jnp.ones((1000,))*2
test_data20 = jnp.stack([time, theta, phi], axis=-1)

time = jnp.ones((1000,))*2.1
test_data21 = jnp.stack([time, theta, phi], axis=-1)




r0 = vmap(ricci_scalar , in_axes=(0,None,None))(test_data0,params,metric_matrix)
r1 = vmap(ricci_scalar , in_axes=(0,None,None))(test_data1,params,metric_matrix)
r5 = vmap(ricci_scalar , in_axes=(0,None,None))(test_data5,params,metric_matrix)
r7 = vmap(ricci_scalar , in_axes=(0,None,None))(test_data7,params,metric_matrix)

r15 = vmap(ricci_scalar , in_axes=(0,None,None))(test_data15,params,metric_matrix)
r13 = vmap(ricci_scalar , in_axes=(0,None,None))(test_data13,params,metric_matrix)
r20= vmap(ricci_scalar , in_axes=(0,None,None))(test_data20,params,metric_matrix)

r21= vmap(ricci_scalar , in_axes=(0,None,None))(test_data21,params,metric_matrix)

#r0 = ricci_scalar(test_data0 , params, metric_matrix)


exact_r = (2*jnp.cos(phi))/(2+jnp.cos(phi))
for i in range(10):
    plt.plot(phi,0.1*i*exact_r,'.',label = f'exact {i*0.1}')




plt.plot(test_data0[:,2],r0,'.',label = 't=0')

plt.plot(test_data5[:,2],r5,'.',label = 't=0.5')
#plt.plot(test_data7[:,2],r7,'o',label = 't=0.7')
plt.plot(test_data1[:,2],r1,'.',label = 't=1')
#plt.plot(test_data13[:,2],r13,'o',label = 't=1.3')
#plt.plot(test_data15[:,2],r15,'o',label = 't=1.5')

plt.plot(test_data20[:,2],r20,'.',label = 't=2')

#plt.plot(test_data21[:,2],r21,'o',label = 't=2.1')

#plt.title('Ricci scalar against y for multiple times')
plt.xlabel('y')
plt.ylabel('Ricci scalar')
plt.legend()
plt.show()


plt.plot(hist,'o')
plt.show()

exact_r = (2*jnp.cos(phi))/(2+jnp.cos(phi))
mse = jnp.mean(jnp.square(exact_r-r0))
print('mse',mse)
