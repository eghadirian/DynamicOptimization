from rockit import *
import matplotlib.pyplot as plt

ocp = Ocp(T=10)
x1 = ocp.state()
x2 = ocp.state()
u = ocp.control(order=0)
p = ocp.control(order=1)
z = ocp.algebraic()

ocp.set_der(x1, x1 - z*x2**2 + u)
ocp.set_der(x2, x1+p)
ocp.add_alg(z-(1 - x1**3)**2)

ocp.add_objective(ocp.integral(x2**2 + x1**2 + u**2 - p**2 + z**2))

ocp.subject_to(-1 <= (u <= 1))
ocp.subject_to(-0.2 <= (u*p <= 0.2))
ocp.subject_to(-0.25 <= (x1 <= 1))
ocp.subject_to(-0.3 <=(ocp.der(p)<=0.3))
ocp.subject_to(ocp.at_t0(x1) == 0)
ocp.subject_to(ocp.at_tf(x2) == 1)

ocp.solver('ipopt')
method = DirectCollocation(N=20)
ocp.method(method)
sol = ocp.solve()

tsc, x1c = sol.sample(x1, grid='integrator')
tsc, x2c = sol.sample(x2, grid='integrator')
tsc, uc = sol.sample(u, grid='integrator')
tsc, pc = sol.sample(p, grid='integrator')
plt.plot(tsc, x1c, '-o', label='state 1')
plt.plot(tsc, x2c, '-^', label='state 2')
plt.plot(tsc, uc, '-.', label='control 1')
plt.plot(tsc, pc, '--', label='control 2')
plt.xlabel('time, s')
plt.legend()
plt.show()

