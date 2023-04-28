import torch
import torch.nn as nn

import torch

def eclamp(x, lower, upper):
    # In-place!!
    if type(lower) == type(x):
        assert x.size() == lower.size()

    if type(upper) == type(x):
        assert x.size() == upper.size()

    I = x < lower
    x[I] = lower[I] if not isinstance(lower, float) else lower

    I = x > upper
    x[I] = upper[I] if not isinstance(upper, float) else upper

    return x

def bger(x, y):
    return x.unsqueeze(2).bmm(y.unsqueeze(1))


# @profile
def pnqp(H, q, lower, upper, x_init=None, n_iter=20):
    GAMMA = 0.1
    n_batch, n, _ = H.size()
    pnqp_I = 1e-11*torch.eye(n).type_as(H).expand_as(H)


    def obj(x):
        return 0.5*bquad(x, H) + bdot(q, x)

    if x_init is None:
        if n == 1:
            x_init = -(1./H.squeeze(2))*q
        else:
            H_lu = H.lu()
            x_init = -q.unsqueeze(2).lu_solve(*H_lu).squeeze(2) # Clamped in the x assignment.
    else:
        x_init = x_init.clone() # Don't over-write the original x_init.

    x = eclamp(x_init, lower, upper)

    # Active examples in the batch.
    J = torch.ones(n_batch).type_as(x).byte()

    for i in range(n_iter):
        g = bmv(H, x) + q

        # TODO: Could clean up the types here.
        Ic = (((x == lower) & (g > 0)) | ((x == upper) & (g < 0))).float()
        If = 1-Ic

        if If.is_cuda:
            Hff_I = bger(If.float(), If.float()).type_as(If)
            not_Hff_I = 1-Hff_I
            Hfc_I = bger(If.float(), Ic.float()).type_as(If)
        else:
            Hff_I = bger(If, If)
            not_Hff_I = 1-Hff_I
            Hfc_I = bger(If, Ic)

        g_ = g.clone()
        g_[Ic.bool()] = 0.
        H_ = H.clone()
        H_[not_Hff_I.bool()] = 0.0
        H_ += pnqp_I

        if n == 1:
            dx = -(1./H_.squeeze(2))*g_
        else:
            H_lu_ = H_.lu()
            dx = -g_.unsqueeze(2).lu_solve(*H_lu_).squeeze(2)

        J = torch.norm(dx, 2, 1) >= 1e-4
        m = J.sum().item() # Number of active examples in the batch.
        if m == 0:
            return x, H_ if n == 1 else H_lu_, If, i

        alpha = torch.ones(n_batch).type_as(x)
        decay = 0.1
        max_armijo = GAMMA
        count = 0
        while max_armijo <= GAMMA and count < 10:
            # Crude way of making sure too much time isn't being spent
            # doing the line search.
            # assert count < 10

            maybe_x = eclamp(x+torch.diag(alpha).mm(dx), lower, upper)
            armijos = (GAMMA+1e-6)*torch.ones(n_batch).type_as(x)
            armijos[J] = (obj(x)-obj(maybe_x))[J]/bdot(g, x-maybe_x)[J]
            I = armijos <= GAMMA
            alpha[I] *= decay
            max_armijo = torch.max(armijos)
            count += 1

        x = maybe_x

    return x, H_ if n == 1 else H_lu_, If, i



def bquad(x, Q):
    return x.unsqueeze(1).bmm(Q).bmm(x.unsqueeze(2)).squeeze(1).squeeze(1)

def bmv(X,y):

    return X.bmm(y.unsqueeze(2)).squeeze(2)


def bdot(x, y):
    return torch.bmm(x.unsqueeze(1), y.unsqueeze(2)).squeeze(1).squeeze(1)

class LQR_solver():
    def __init__(self,n_state,n_ctrl,T,lqr_iter=10,u_lower=None,u_upper=None):

        self.lqr_iter=lqr_iter

        self.n_state=n_state

        self.n_ctrl=n_ctrl

        self.T=T

        self.max_linesearch_iter=10

        self.linesearch_decay=0.2

        self.u_lower = u_lower

        self.u_upper = u_upper

        self.eps = 1e-7

        self.best_cost_eps = 1e-4

        self.not_improved_lim=5

    def get_traj(self,x_init,u,F):
        x = [x_init]
        for t in range(self.T):
            xt = x[t]
            ut = u[t]
            if t < self.T - 1:
                xut = torch.cat((xt, ut), 1)

                new_x =  bmv(F[t], xut)

                x.append(new_x)
        x = torch.stack(x, dim=0)

        return x

    def get_cost(self,x,u,C,c):
        objs = []
        for t in range(self.T):
            xt = x[t]
            ut = u[t]
            xut = torch.cat((xt, ut), 1)
            obj = 0.5 * bquad(xut, C[t]) + bdot(xut, c[t])
            objs.append(obj)
        objs = torch.stack(objs, dim=0)
        total_obj = torch.sum(objs, dim=0)

        return total_obj

    def lqr_backward(self,u, C, c, F, f):
        Ks = []
        ks = []
        Vtp1 = vtp1 = None
        prev_kt=None

        for t in range(self.T-1, -1, -1):
            if t == self.T-1:
                Qt = C[t]
                qt = c[t]
            else:
                Ft = F[t]
                Ft_T = Ft.transpose(1,2)
                Qt = C[t] + Ft_T.bmm(Vtp1).bmm(Ft)
                if f is None or f.nelement() == 0:
                    qt = c[t] + Ft_T.bmm(vtp1.unsqueeze(2)).squeeze(2)
                else:
                    ft = f[t]
                    qt = c[t] + Ft_T.bmm(Vtp1).bmm(ft.unsqueeze(2)).squeeze(2) + \
                        Ft_T.bmm(vtp1.unsqueeze(2)).squeeze(2)

            Qt_xx = Qt[:, :self.n_state, :self.n_state]
            Qt_xu = Qt[:, :self.n_state, self.n_state:]
            Qt_ux = Qt[:, self.n_state:, :self.n_state]
            Qt_uu = Qt[:, self.n_state:, self.n_state:]
            qt_x = qt[:, :self.n_state]
            qt_u = qt[:, self.n_state:]

            if self.u_lower is None:
                Qt_uu_inv =torch.pinverse(Qt_uu)
                Kt = -Qt_uu_inv.bmm(Qt_ux)
                kt = bmv(-Qt_uu_inv, qt_u)

            else:
                lb = self.u_lower - u[t]
                ub = self.u_upper - u[t]
                kt, Qt_uu_free_LU, If, n_qp_iter = pnqp(
                    Qt_uu, qt_u, lb, ub,
                    x_init=prev_kt, n_iter=20)

                prev_kt = kt
                Qt_ux_ = Qt_ux.clone()
                Qt_ux_[(1-If).unsqueeze(2).repeat(1,1,Qt_ux.size(2)).bool()] = 0
                Kt = -Qt_ux_.lu_solve(*Qt_uu_free_LU)

            Kt_T = Kt.transpose(1, 2)

            Ks.append(Kt)
            ks.append(kt)


            Vtp1 = Qt_xx + Qt_xu.bmm(Kt) + Kt_T.bmm(Qt_ux) + Kt_T.bmm(Qt_uu).bmm(Kt)
            vtp1 = qt_x + Qt_xu.bmm(kt.unsqueeze(2)).squeeze(2) + \
                   Kt_T.bmm(qt_u.unsqueeze(2)).squeeze(2) + \
                   Kt_T.bmm(Qt_uu).bmm(kt.unsqueeze(2)).squeeze(2)

        return Ks, ks

    def lqr_forward(self,x,u,x_init,C,c,F,Ks, ks):
        n_batch = C.size(1)

        old_cost = self.get_cost(x,u,C,c)

        current_cost = None
        alphas = torch.ones(n_batch).type_as(C)

        i = 0

        full_du_norm=None


        while (current_cost is None or  (old_cost is not None and  torch.any((current_cost > old_cost)).cpu().item() == 1)) and \
            i < self.max_linesearch_iter:
            new_u = []
            new_x = [x_init]
            dx = [torch.zeros_like(x_init)]
            objs = []
            for t in range(self.T):
                t_rev = self.T-1-t
                Kt = Ks[t_rev]
                kt = ks[t_rev]
                new_xt = new_x[t]
                xt = x[t]
                ut = u[t]
                dxt = dx[t]

                new_ut = bmv(Kt, dxt) + ut + torch.diag(alphas).mm(kt)

                if self.u_lower is not None:
                    new_ut = torch.clamp(new_ut, min=self.u_lower, max=self.u_upper)

                new_u.append(new_ut)

                new_xut = torch.cat((new_xt, new_ut), dim=1)

                if t < self.T-1:
                    new_xtp1 = bmv(F[t], new_xut)

                    new_x.append(new_xtp1)
                    dx.append(new_xtp1 - x[t+1])

                obj = 0.5 * bquad(new_xut, C[t]) + bdot(new_xut, c[t])

                objs.append(obj)

            objs = torch.stack(objs)
            current_cost = torch.sum(objs, dim=0)

            new_u = torch.stack(new_u)
            new_x = torch.stack(new_x)

            full_du_norm = (u - new_u).transpose(1, 2).contiguous().view(
                n_batch, -1).norm(2, 1)

            alphas[current_cost > old_cost] *= self.linesearch_decay
            i += 1

        return new_x,new_u,full_du_norm,current_cost

    def solve_lqr(self,x,u,x_init,C,c,F):
        c_back = []
        for t in range(self.T):
            xt = x[t]
            ut = u[t]
            xut = torch.cat((xt, ut), 1)
            c_back.append(bmv(C[t], xut) + c[t])
        c_back = torch.stack(c_back)
        f_back = None

        Ks, ks = self.lqr_backward(u, C, c_back, F, f_back)
        new_x, new_u, full_du_norm,current_cost = self.lqr_forward(x, u, x_init, C, c, F, Ks, ks)

        return new_x,new_u,full_du_norm,current_cost

    def solve(self, x_init,C,c,F):
        n_batch=x_init.shape[0]

        u = torch.zeros(self.T, n_batch, self.n_ctrl).to(x_init.device)

        best=None
        n_not_improved = 0

        for i in range(self.lqr_iter):

            x=self.get_traj(x_init,u,F)

            x,u,full_du_norm,costs=self.solve_lqr(x,u,x_init,C,c,F)

            n_not_improved += 1

            if best is None:
                best = {
                    'x': list(torch.split(x, split_size_or_sections=1, dim=1)),
                    'costs': costs,
                }
            else:
                for j in range(n_batch):
                    if costs[j] <= best['costs'][j] + self.best_cost_eps:
                        n_not_improved = 0
                        best['x'][j] = x[:, j].unsqueeze(1)
                        best['costs'][j] = costs[j]

            if max(full_du_norm) < self.eps or  n_not_improved > self.not_improved_lim:
                break

        x = torch.cat(best['x'], dim=1)

        return x








