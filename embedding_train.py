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

####### Defining Functions

class LearnedMetricJAV(nn.Module):
    """
    Parameters
    ----------
        dim         : Dimension of manifold/input.
        n_units     : Number of units in each layer.
        activation  : Nonlinearity between each layer.
    """
    dim: int
    metric: False

    def setup(self):

        if self.metric == True:
            self.n_units: Sequence[int] = (16, 32, 16)
            self.activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.softplus
        else:
            self.n_units: Sequence[int] = (16, 32, 16)
            self.activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.softplus
        self.n_hidden = len(self.n_units)
        self.layers = [nn.Dense(f) for f in self.n_units]

    @nn.compact
    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != self.n_hidden - 1:
                x = self.activation(x)

        # L_x = nn.Dense(self.dim * (self.dim - 1) // 2, name='L')(x)
        if self.metric == True:
            D_x = nn.Dense(self.dim + 1, name='D')(x)
        else:
            D_x = nn.Dense(self.dim + 1, name='D')(x)
        D_x = self.activation(D_x)

        # jnp.diag(D_x) for matrix D
        # utils.fill_lower_tri(L_x, self.dim) for matrix L
        return D_x


def create_train_state(rng, model, optimizer, metric):
    rng, init_rng = jax.random.split(rng)
    if metric:
        params = model.init(rng, jnp.ones([1, 3]))['params']
    else:
        params = model.init(rng, jnp.ones([1, 2]))['params']
    opt_state = optimizer.init(params)
    return params, opt_state, init_rng


def T2_simple_collocation(key, n):
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
    key, key_phi, key_theta, key_time = jax.random.split(key, 4)
    phi = jax.random.uniform(key_phi, shape=(n,)) * 2 * jnp.pi
    theta = jax.random.uniform(key_theta, shape=(n,)) * 2 * jnp.pi
    time = jax.random.uniform(key_time, shape=(n,)) * 2
    return jnp.stack([time, theta, phi], axis=-1)


def xyz_collocation(key, n):
    key, key_phi, key_theta, key_time = jax.random.split(key, 4)
    phi = jax.random.uniform(key_phi, shape=(n,)) * 2 * jnp.pi
    theta = jax.random.uniform(key_theta, shape=(n,)) * 2 * jnp.pi
    # time = (jax.random.uniform(key_time, shape=(n,))*0)
    time = jnp.ones((n,)) * 0
    return jnp.stack([theta, phi], axis=-1), jnp.stack([time, theta, phi], axis=-1)


def T2_simple_initial(key, n):
    key, key_phi, key_theta, key_time = jax.random.split(key, 4)
    time = jax.random.uniform(key_time, shape=(n,)) * 0
    phi = jax.random.uniform(key_phi, shape=(n,)) * 2 * jnp.pi
    theta = jax.random.uniform(key_theta, shape=(n,)) * 2 * jnp.pi
    return jnp.stack([time, theta, phi], axis=-1)


def T2_g_0_generator(n, Data_i):
    theta = Data_i[:, 1]
    phi = Data_i[:, 2]

    g11 = (2 + jnp.cos(phi)) ** 2
    g12 = jnp.zeros((n,))
    g21 = jnp.zeros((n,))
    g22 = jnp.ones((n,))
    return jnp.stack([g11, g12, g21, g22], axis=-1).reshape(n, 2, 2)

def metric_matrixJAV(p, params):
    return LearnedMetricJAV(p.shape[-1], True).apply({'params': params}, p) #.reshape(32,2,2)

def get_metricJAV(p, params , metric_out):
    D = metric_out(p,params)
    g = D.reshape(2,2)
    gT = jnp.transpose(g,(1,0))
    return jnp.matmul(gT,g)


def inverse_metric_matrixJAV(p, params, metric_out):
    g = get_metricJAV(p, params, metric_out)
    return jnp.linalg.inv(g)


def dgdtJAV(p, params, metric_out):
    derivatives = jacfwd(get_metricJAV, argnums=0)(p, params, metric_out)  # [..., i,j,l]
    g_t = derivatives[:, :, 0]
    return g_t


def christoffel_symbolsJAV(p, params, metric_out):
    g_inv = inverse_metric_matrixJAV(p, params, metric_out)
    derivatives = jacfwd(get_metricJAV, argnums=0)(p, params, metric_out)  # [..., i,j,l]
    g_t = derivatives[:, :, 0]
    jac_g = derivatives[:, :, 1:]

    x1 = jnp.einsum('...kl, ...jli->...kij', g_inv, jac_g)
    x2 = jnp.einsum('...kl, ...ilj->...kij', g_inv, jac_g)
    x3 = jnp.einsum('...kl, ...ijl->...kij', g_inv, jac_g)

    cs = 0.5 * (x1 + x2 - x3)
    return cs


def riemann_curvature_tensorJAV(p, params, metric_out):
    # [..., i,j,k,l]
    derivatives_cs = jacfwd(christoffel_symbolsJAV, argnums=0)(p, params, metric_out)
    jac_cs = derivatives_cs[:, :, :, 1:]
    cs = christoffel_symbolsJAV(p, params, metric_out)

    x1 = jnp.einsum('...iljk->...ijkl', jac_cs)
    x2 = jnp.einsum('...ikjl->...ijkl', jac_cs)
    x3 = jnp.einsum('...ikm, ...mlj->...ijkl', cs, cs)
    x4 = jnp.einsum('...ilm, ...mkj->...ijkl', cs, cs)

    riemann = x1 - x2 + x3 - x4
    return riemann


def ricci_tensorJAV(p, params, metric_out):
    riemann = riemann_curvature_tensorJAV(p, params, metric_out)
    ricci = jnp.einsum('...kikj->...ij', riemann)
    # print(ricci.shape)
    return ricci


def ricci_scalarJAV(p, params, metric_out):
    g_inv = inverse_metric_matrixJAV(p, params, metric_out)
    ricci_t = ricci_tensorJAV(p, params, metric_out)
    R = jnp.einsum('...ij, ...ij->...', g_inv, ricci_t)
    return R


def residual_loss(p, params, metric_out):
    dgdt = vmap(dgdtJAV, in_axes=(0, None, None))(p, params, metric_out)
    ricci = vmap(ricci_tensorJAV, in_axes=(0, None, None))(p, params, metric_out)
    a = jnp.square(dgdt + 2 * ricci)
    b = jnp.mean(a)
    return b


def initial_loss(n, params, metric_out, Data_i):
    g = vmap(get_metricJAV, in_axes=(0, None, None))(Data_i, params, metric_out)

    g_0 = T2_g_0_generator(n, Data_i)
    a = jnp.square(g - g_0)
    b = jnp.mean(a)
    return b


def periodic_loss(n, params, metric_out, Data_r):
    g = vmap(get_metricJAV, in_axes=(0, None, None))(Data_r, params, metric_out)
    Data_r_transformed = periodic_transformer(Data_r)
    g_transformed = vmap(get_metricJAV, in_axes=(0, None, None))(Data_r_transformed, params, metric_out)
    a = jnp.square(g - g_transformed)
    b = jnp.mean(a)
    return b


def periodic_transformer(Data):
    time = Data[:, 0]
    theta = Data[:, 1]
    phi = Data[:, 2]
    theta_transformed = 2 * jnp.pi - theta
    phi_transformed = 2 * jnp.pi - phi
    return jnp.stack([time, theta_transformed, phi_transformed], axis=-1)


def metric_output(p, params, metric):
    return LearnedMetricJAV(2, metric).apply({'params': params}, p)  # .reshape(32,2,2)


def derivatives(p, params, out, metric):
    derivative_array = jacfwd(out, argnums=0)(p, params, False)
    # print(derivative_array)
    return derivative_array


def edge_data_generator(p, variable, value):
    u = p[:, 0]
    v = p[:, 1]
    if variable == 0:
        u = (u * 0 + 1) * value
        return jnp.stack([u, v], axis=-1)
    else:
        v = (v * 0 + 1) * value
        return jnp.stack([u, v], axis=-1)


def u_rot_data(p, key):
    u = p[:, 0]
    v = p[:, 1]
    n = u.shape[-1]
    u = u + jax.random.uniform(key, shape=(n,)) * 2 * jnp.pi
    u = u % (2 * jnp.pi)
    return jnp.stack([u, v], axis=-1)


def residual_loss_xyz(p, params_x, metric_output, metric, p_metric, params_metric, key):
    # print('pre')
    key, key_rot = jax.random.split(key)

    grad_x = vmap(derivatives, in_axes=(0, None, None, None))(p, params_x, metric_output, False)
    #generate transformed data
    p_u_0 = edge_data_generator(p, 0, 0.0)
    p_u_2p = edge_data_generator(p, 0, 2 * jnp.pi)
    p_v_0 = edge_data_generator(p, 1, 0.0)
    p_v_2p = edge_data_generator(p, 1, 2 * jnp.pi)

    #gradient matching terms
    grad_x_u_0 = vmap(derivatives, in_axes=(0, None, None, None))(p_u_0, params_x, metric_output, False)
    grad_x_u_2p = vmap(derivatives, in_axes=(0, None, None, None))(p_u_2p, params_x, metric_output, False)
    grad_x_v_0 = vmap(derivatives, in_axes=(0, None, None, None))(p_v_0, params_x, metric_output, False)
    grad_x_v_2p = vmap(derivatives, in_axes=(0, None, None, None))(p_v_2p, params_x, metric_output, False)

    xyz_u_0 = vmap(metric_output, in_axes=(0, None, None))(p_u_0, params_x, False)
    xyz_u_2p = vmap(metric_output, in_axes=(0, None, None))(p_u_2p, params_x, False)
    xyz_v_0 = vmap(metric_output, in_axes=(0, None, None))(p_v_0, params_x, False)
    xyz_v_2p = vmap(metric_output, in_axes=(0, None, None))(p_v_2p, params_x, False)

    loss_grad_edge = 10 * jnp.mean(jnp.square(grad_x_u_0 - grad_x_u_2p)) + 10 * jnp.mean(
        jnp.square(grad_x_v_0 - grad_x_v_2p))
    loss_edge = 10 * jnp.mean(jnp.square(xyz_u_0 - xyz_u_2p)) + 10 * jnp.mean(jnp.square(xyz_v_0 - xyz_v_2p))

    g = vmap(get_metricJAV, in_axes=(0, None, None))(p_metric, params_metric, metric_matrixJAV)

    dxdu = grad_x[:, 0, 0]
    dxdv = grad_x[:, 0, 1]
    dydu = grad_x[:, 1, 0]
    dydv = grad_x[:, 1, 1]
    dzdu = grad_x[:, 2, 0]
    dzdv = grad_x[:, 2, 1]

    g_uu = g[:, 0, 0]
    g_uv = g[:, 0, 1]
    g_vv = g[:, 1, 1]

    l1 = jnp.mean(jnp.square(dxdu ** 2 + dydu ** 2 + dzdu ** 2 - g_uu))
    l2 = jnp.mean(jnp.square(dxdu * dxdv + dydu * dydv + dzdu * dzdv - g_uv))
    l3 = jnp.mean(jnp.square(dxdv ** 2 + dydv ** 2 + dzdv ** 2 - g_vv))

    p_trans_u = transform_data(p, 0, key=key)
    p_trans_v = transform_data(p, 1, key=key)

    p_trans_u_quarter = transform_data(p, 0, 0, key=key)

    p_v_reflection = transform_data(p, 1, 1, key=key)

    p_trans_u_delta = transform_data(p, 0, 2, key=key)

    xyz = vmap(metric_output, in_axes=(0, None, None))(p, params_x, False)
    xyz_u_trans = vmap(metric_output, in_axes=(0, None, None))(p_trans_u, params_x, False)
    xyz_v_trans = vmap(metric_output, in_axes=(0, None, None))(p_trans_v, params_x, False)

    xyz_u_trans_quarter = vmap(metric_output, in_axes=(0, None, None))(p_trans_u_quarter, params_x, False)
    xyz_v_reflection = vmap(metric_output, in_axes=(0, None, None))(p_v_reflection, params_x, False)
    xyz_u_trans_delta = vmap(metric_output, in_axes=(0, None, None))(p_trans_u_delta, params_x, False)

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    p_u_rot = u_rot_data(p, key_rot)
    rotated_xyz = vmap(metric_output, in_axes=(0, None, None))(p_u_rot, params_x, False)
    x_rot = rotated_xyz[:, 0]
    y_rot = rotated_xyz[:, 1]
    z_rot = rotated_xyz[:, 2]
    radius_rot = x_rot ** 2 + y_rot ** 2
    radius = x ** 2 + y ** 2
    loss_u_rot_radius = jnp.mean(jnp.square(radius_rot - radius))
    loss_u_rot_z = jnp.mean(jnp.square(z - z_rot))
    loss_u_rot = loss_u_rot_z + loss_u_rot_radius

    x_u_trans = xyz_u_trans[:, 0]
    x_v_trans = xyz_v_trans[:, 0]

    y_u_trans_quarter = xyz_u_trans_quarter[:, 1]
    y_v_trans = xyz_v_trans[:, 1]

    z_u_trans = xyz_u_trans_delta[:, 2]
    z_v_trans_quarter = xyz_v_reflection[:, 2]

    tx1 = jnp.mean(jnp.square(x - x_u_trans))
    tx2 = jnp.mean(jnp.square(x - x_v_trans))

    ty1 = jnp.mean(jnp.square(y - y_u_trans_quarter))
    ty2 = jnp.mean(jnp.square(y - y_v_trans))

    tz1 = jnp.mean(jnp.square(z - z_u_trans))
    tz2 = jnp.mean(jnp.square(jnp.abs(z) - jnp.abs(z_v_trans_quarter)))

    t = (tx1 + tx2) + (ty1 + ty2) + (tz1 + tz2)

    l = 4 * (l1 + l2 + l3) + t + loss_grad_edge + loss_edge  # + loss_u_rot
    return l


def transform_data(data, variable, quarter=10, key=1):
    u = data[:, 0]
    v = data[:, 1]

    if quarter == 1 and variable == 1:
        v = 2 * jnp.pi - v
        return jnp.stack([u, v], axis=-1)

    if quarter == 0:
        u = jnp.pi / 2.0 + u
        u = u % (2 * jnp.pi)
    elif quarter == 1:
        v = jnp.pi / 2.0 + v
        v = v % (2 * jnp.pi)
    elif quarter == 2:
        n = u.shape[-1]
        u = u + jax.random.uniform(key, shape=(n,))
        u = u % (2 * jnp.pi)
        return jnp.stack([u, v], axis=-1)
    if variable == 0:
        u = 2 * jnp.pi - u
    else:
        v = 2 * jnp.pi - v
    return jnp.stack([u, v], axis=-1)

def train_metric_gJAV(params_g, opt_state_g, optimizer_g, data_r, data_i, counter, hist):
    # f, grad_f = vmap(jax.value_and_grad(scalar_f, argnums=0), in_axes=(0, None))(data, params_f)

    grads_g = jax.grad(loss, argnums=2)(data_r, data_i, params_g, metric_matrixJAV, 1000)
    param_updates_g, opt_state_g = optimizer_g.update(grads_g, opt_state_g, params_g)
    params_g = optax.apply_updates(params_g, param_updates_g)
    if counter % 20 == 0:
        l = loss(data_r, data_i, params_g, metric_matrixJAV, 1000)
        print(l)
        hist.append(l)
    return params_g, opt_state_g


def train_NN(params_x,params_metric,opt_state_x,optimizer_x,data_r,data_r_metric,counter,hist,key):
    #params, opt_state, optimizer, data_r, counter ):
    #f, grad_f = vmap(jax.value_and_grad(scalar_f, argnums=0), in_axes=(0, None))(data, params_f)

    grads_x = jax.grad(residual_loss_xyz, argnums=1)(data_r,params_x,metric_output,False,data_r_metric,params_metric,key)
    #grads_y = jax.grad(residual_loss_xyz, argnums=2)(data_r,params_x,params_y,params_z,metric_output,False,data_r_metric,params_metric)
    #grads_z = jax.grad(residual_loss_xyz, argnums=3)(data_r,params_x,params_y,params_z,metric_output,False,data_r_metric,params_metric)

    param_updates_x , opt_state_x = optimizer_x.update(grads_x,opt_state_x,params_x)
    #param_updates_y , opt_state_y = optimizer_y.update(grads_y,opt_state_y,params_y)
    #param_updates_z , opt_state_z = optimizer_z.update(grads_z,opt_state_z,params_z)

    #params_g = optax.apply_updates(params_g, param_updates_g)

    params_x = optax.apply_updates(params_x , param_updates_x)
    #params_y = optax.apply_updates(params_y , param_updates_y)
    #params_z = optax.apply_updates(params_z , param_updates_z)
    if counter % 20 == 0:
      l = residual_loss_xyz(data_r, params_x,metric_output,False , data_r_metric , params_metric,key)
      print('loss:',l)
      print('iteration:',counter)
      print('#########')
      hist.append(l)


    counter += 1
    return params_x, opt_state_x

import time

###### Pre-Training

collocation_point_sampler= partial(xyz_collocation , n = 1000)
seed = int(time.time())  # 42
rng = jax.random.PRNGKey(seed)
rng, init_rng_x , init_rng_y, init_rng_z ,init_rng_metric = jax.random.split(rng,5)

optimizer_x = optax.adam(learning_rate = 3e-4)
optimizer_metric = optax.adam(learning_rate = 3e-4)


x = LearnedMetricJAV(2,False)
g = LearnedMetricJAV(3,True)

params_x, opt_state_x, init_rng_x = create_train_state(init_rng_x, x,  optimizer_x , False)
params_metric, opt_state_metric, init_rng_metric = create_train_state(init_rng_metric, g,  optimizer_metric,True)

params_metric = pickle.load(open('param_norm.pkl','rb'))
optimizer_metric = pickle.load(open('opt_state_norm.pkl','rb'))

params_x = pickle.load(open('params_xyz_norm.pkl','rb'))
opt_state_x = pickle.load(open('opt_state_xyz_norm.pkl','rb'))

rng, sample_rng_col , sample_rng_initial = jax.random.split(rng , 3)
Data_r  , Data_r_metric = collocation_point_sampler(key = sample_rng_col)

metric = vmap(metric_output, in_axes=(0,None,None))(Data_r, params_x,False)#metric_output(Data_r , params_x)
grad_x = vmap(derivatives , in_axes=(0,None,None,None))(Data_r, params_x, metric_output,False)

###### Training

t0 = time.time()
counter = 0
hist = []
for t in range(int(15000)):

    if t % 20 == 0:
        print(f'Iteration {t} =====>>>')
        # if t % 100 == 0:
        # pickle.dump(params_x, open('drive/MyDrive/AarjavRICCI/params_t2_increment3 (1) (1) (1).pkl','wb'))
        # pickle.dump(opt_state_x, open('drive/MyDrive/AarjavRICCI/opt_state_t2_increment3 (1) (1) (1).pkl','wb'))

        pickle.dump(params_x, open('params_xyz_norm.pkl', 'wb'))
        pickle.dump(opt_state_x, open('opt_state_xyz_norm.pkl', 'wb'))
    rng, sample_rng_col, sample_rng_initial, random_delta = jax.random.split(rng, 4)
    Data_r, Data_r_metric = collocation_point_sampler(key=sample_rng_col)
    # Data_i = initial_point_sampler(key = sample_rng_initial)
    params_x, opt_state_x = train_NN(params_x, params_metric, opt_state_x, optimizer_x, Data_r, Data_r_metric, t, hist,
                                     random_delta)
print(time.time()-t0)
plt.plot(np.log(hist[0:]))
plt.show()


######### PLOTS ############
seed = int(3)  #time.time())  # 42
key = jax.random.PRNGKey(seed)
key, key_phi, key_theta , key_time = jax.random.split(key, 4)

#generate data points
phi_rand = jax.random.uniform(key_time, shape=(1000,))*2*jnp.pi
theta_const = jax.random.uniform(key_phi, shape=(1000,)) * 0
test_data_phi = jnp.stack([theta_const, phi_rand], axis=-1)
theta_rand = jax.random.uniform(key_time, shape=(1000,))*2*jnp.pi
phi_const = jax.random.uniform(key_phi, shape=(1000,)) * 0
test_data_theta = jnp.stack([theta_rand, phi_const], axis=-1)

theta_rand = jax.random.uniform(key_time, shape=(1000,))*2*jnp.pi
phi_rand = jax.random.uniform(key_phi, shape=(1000,)) *2*jnp.pi
test_data = jnp.stack([theta_rand, phi_rand], axis=-1)
xyz = vmap(metric_output , in_axes=(0,None,None))(test_data, params_x,False)
x = xyz[:,0]
y = xyz[:,1]
z = xyz[:,2]

ax = plt.axes(projection='3d')

# Data for a three-dimensional line

ax.scatter3D(x, y, z, 'gray')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


xyz_u = vmap(metric_output , in_axes=(0,None,None))(test_data_theta, params_x,False)
xyz_v = vmap(metric_output , in_axes=(0,None,None))(test_data_phi, params_x,False)
theta_const = jax.random.uniform(key_theta, shape=(1000,)) * 0
phi_const = jax.random.uniform(key_phi, shape=(1000,)) * 0
x_u = xyz_u[:,0]
y_u = xyz_u[:,1]
z_u = xyz_u[:,2]


x_v = xyz_v[:,0]
y_v = xyz_v[:,1]
z_v = xyz_v[:,2]

fig, axs = plt.subplots(2, 3,figsize=(10, 5),sharey='col')
#fig = plt.figure(figsize=(6, 9))
axs[0, 0].plot(theta_rand, x_u,'.')
axs[0, 0].set_title('x(u)')
axs[0, 1].plot(theta_rand, y_u,'.')
axs[0, 1].set_title('y(u)')
axs[0, 2].plot(theta_rand, z_u,'.')
axs[0, 2].set_title('z(u)')
axs[1, 0].plot(phi_rand, x_v,'.')
axs[1, 0].set_title('x(v)')
axs[1, 1].plot(phi_rand, y_v,'.')
axs[1, 1].set_title('y(v)')
axs[1, 2].plot(phi_rand, z_v,'.')
axs[1, 2].set_title('z(v)')
#axs[1,2].set_zlim3d(0, 1)
fig.tight_layout()
plt.show()
