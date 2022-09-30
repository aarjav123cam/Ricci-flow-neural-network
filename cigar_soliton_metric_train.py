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
import seaborn as sns
sns.set()


###################### Defining functions ###########################

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

def cigar_simple_collocation(key , n ):
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
    phi = jax.random.uniform(key_phi, shape=(n,)) * 10
    theta = jax.random.uniform(key_theta, shape=(n,)) * 10
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

def cigar_g_0_generator(n, Data_i):
  theta = Data_i[:, 1]
  phi = Data_i[:, 2]

  g11 = (1 + phi**2+theta**2)**-1
  g12 = jnp.zeros((n,))
  g21 = jnp.zeros((n,))
  g22 = (1 + phi**2+theta**2)**-1

  return jnp.stack([g11, g12, g21,g22], axis=-1).reshape(n,2,2)

def metric_matrix(p, params):
    return LearnedMetric(p.shape[-1], (16, 32, 16)).apply({'params': params}, p) #.reshape(32,2,2)

def get_metric(p, params , metric_out):
    D = metric_out(p,params)
    g = D.reshape(2,2)
    gT = jnp.transpose(g,(1,0))
    return jnp.matmul(gT,g)

def inverse_metric_matrix(p, params, metric_out):
    g = get_metric(p, params, metric_out)
    g_inv = jnp.linalg.inv(g)
    return jnp.linalg.inv(g)

def dgdt(p,params,metric_out):
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

def residual_loss(p, params, metric_out):
    dgdt = vmap(dgdt, in_axes=(0, None, None))(p, params, metric_out)
    ricci = vmap(ricci_tensor, in_axes=(0, None, None))(p, params, metric_out)
    a = jnp.square(dgdt + 2 * ricci)
    b = jnp.mean(a)
    return b

def initial_loss(n, params, metric_out, Data_i):
  #key = jax.random.PRNGKey(2)
  #Data_i = T2_simple_initial(key , n)
  g = vmap(get_metric, in_axes=(0,None,None))(Data_i, params, metric_out)

  g_0 = cigar_g_0_generator(n,Data_i)
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

def rotation_loss(n,params,metric_out,Data_r,key):

    g = vmap(get_metric, in_axes=(0,None,None))(Data_r, params, metric_out)
    Data_r_transformed = rotation_transformer(Data_r,key,n)
    g_transformed = vmap(get_metric, in_axes=(0,None,None))(Data_r_transformed, params, metric_out)
    a = jnp.square(g - g_transformed)
    b = jnp.mean(a)
    return  b

def rotation_transformer(Data,key,n):
    time = Data[:,0]
    theta = Data[:,1]
    phi = Data[:,2]
    rotation_angle =  jax.random.uniform(key, shape=(n,)) * 2 * jnp.pi
    theta_transformed = theta*jnp.cos(rotation_angle) - phi*jnp.sin(rotation_angle)
    phi_transformed = theta*jnp.sin(rotation_angle) + phi*jnp.cos(rotation_angle)
    return jnp.stack([time, theta_transformed, phi_transformed], axis=-1)

def loss(Data_r,Data_i,params,metric_out,n,key):
  loss_res = residual_loss(Data_r,params,metric_out)
  loss_init = initial_loss(n,params,metric_out,Data_i)
  loss_rotation = rotation_loss(n,params , metric_out , Data_r,key)
  l = loss_init + loss_res +  loss_rotation    #loss_periodic
  #print(jnp.array([l]))
  return l

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

###################### Main code ###########################
collocation_point_sampler =  partial(cigar_simple_collocation, n=1000)
initial_point_sampler = partial(cigar_simple_initial,n=1000)

seed = int(time.time())
rng = jax.random.PRNGKey(seed)
rng, init_rng = jax.random.split(rng)
optimizer = optax.adam(learning_rate = 3e-4)
model_g = LearnedMetric(3)
params, opt_state, init_rng = create_train_state(init_rng, model_g,  optimizer)

#params = pickle.load(open('params_cigar_metric.pkl','rb'))
#opt_state = pickle.load(open('opt_state_cigar_metric.pkl','rb'))

hist = []

t0 = time.time()
for t in range(int(20000)):

  if t % 20 == 0:
    print(f'Iteration {t} =====>>>')
  rng, sample_rng_col , sample_rng_initial,rotation_rng = jax.random.split(rng , 4)
  Data_r = collocation_point_sampler(key = sample_rng_col)
  Data_i = initial_point_sampler(key = sample_rng_initial)
  (params, opt_state) = train_metric_g(params,opt_state, optimizer, Data_r,Data_i,t,hist,rotation_rng)
  #if t%100 == 0:
  #    plt.plot(hist)
  #    plt.show()

pickle.dump(params, open('params_cigar_metric.pkl','wb'))
pickle.dump(opt_state, open('opt_state_cigar_metric.pkl','wb'))

print(time.time()-t0)
###################### Plotting ###########################


g0 = vmap(get_metric, in_axes=(0,None,None))(test_data0, params, metric_matrix)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(test_data0[:,1], test_data0[:,2],g0[:,0,0] )
plt.show()





mse_arr = []
mse_norm_arr = []
for i in range(21):
    time = 0.1*i*jnp.ones((1000,))
    testdata = jnp.stack([time, theta, phi], axis=-1)
    g_xx = vmap(get_metricJAV, in_axes=(0, None, None))(test_data0, params, metric_matrixJAV)[:,0,0]
    average_g_xx = jnp.mean(g_xx)
    exact_xx = jnp.sqrt(1/(jnp.exp(4*time)+theta**2+phi**2))
    mse = jnp.mean(jnp.square(g_xx - exact_xx))
    mse_arr.append(mse)
    mse_norm_arr.append(mse/average_g_xx)
times = jnp.linspace(0,2,21)
plt.plot(times,mse_arr)
plt.xlabel('Time after ricci flow',fontsize=20)
plt.ylabel('MSE',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
#plt.plot(mse_norm_arr)
plt.show()