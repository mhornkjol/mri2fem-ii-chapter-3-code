from fenics import *

mu = 8e-4

parameters["krylov_solver"]["monitor_convergence"] = True

mesh = Mesh()
with XDMFFile(mesh.mpi_comm(), "./mesh/brain_mesh_refined.xdmf") as xdmf:
    xdmf.read(mesh)
    subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
    xdmf.read(subdomains)


try:
    with XDMFFile(mesh.mpi_comm(), "./mesh/brain_mesh_refined.xdmf") as xdmf:
        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
        xdmf.read(boundaries)
except RuntimeError:
    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
    with XDMFFile(mesh.mpi_comm(), "./mesh/brain_mesh_refined.xdmf") as xdmf:
        xdmf.read(mvc)#, name="mesh_tags")
    boundaries = MeshFunction("size_t", mesh, mvc)

P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH)

(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

n = FacetNormal(mesh)


class BoundaryAG(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > -31 and x[0] < 18 and x[1] > -65 and x[1] < 13 and x[2] > 60 and on_boundary


Outflow = BoundaryAG()
Outflow.mark(boundaries, 16)

no_slip = Constant((0.0, 0.0, 0.0))
bc1 = DirichletBC(W.sub(0), no_slip, boundaries, 7)
bc2 = DirichletBC(W.sub(0), no_slip, boundaries, 8)
bc3 = DirichletBC(W.sub(0), no_slip, boundaries, 9)
bc4 = DirichletBC(W.sub(0), no_slip, boundaries, 12)
bc5 = DirichletBC(W.sub(0), no_slip, boundaries, 13)

bcs = [bc1, bc2, bc3, bc4, bc5]

dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
dxP = dx((2, 3))
dxF = dx((1, 4, 5, 6))

g_source = Constant(0.006896552)

a = mu * inner(grad(u), grad(v)) * dxF - div(v) * p * dxF - q * div(u) * dxF
p = mu * inner(grad(u), grad(v)) * dxF + (1.0 / mu) * p * q * dxF

L = -g_source * q * dx(6)

A, b = assemble_system(a, L, bcs)
P, _ = assemble_system(p, L, bcs)
A.ident_zeros()
P.ident_zeros()

U = Function(W)
solver = KrylovSolver("minres", "amg")
solver.set_operators(A, P)
it = solver.solve(U.vector(), b)
u, p = U.split(deepcopy=True)
print(f"it {it}")

with XDMFFile(MPI.comm_world, "solution/velocity.xdmf") as xdmf:
    xdmf.write_checkpoint(u, "velocity", 0)

with XDMFFile(MPI.comm_world, "solution/pressure.xdmf") as xdmf:
    xdmf.write_checkpoint(p, "pressure", 0)
