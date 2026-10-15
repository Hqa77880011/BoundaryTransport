import torch
from torch import nn
from torch.nn import functional as F
from .layers import Encoder,classification_demo


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;w=c.get('width',16);classes=c.get('classes',3);shape=c.get('input_shape',[12])
        self.encoder=nn.Sequential(Encoder(shape[0],w),nn.AdaptiveAvgPool2d(1),nn.Flatten()) if len(shape)==3 else nn.Sequential(nn.Linear(shape[-1],w),nn.SiLU(),nn.Linear(w,w))
        self.center=nn.Parameter(torch.randn(classes,w)*.1);self.factor=nn.Parameter(torch.eye(w)[None].repeat(classes,1,1))
        self.radius=nn.Parameter(torch.ones(classes));self.register_buffer('counts',torch.ones(classes));self.register_buffer('dual',torch.tensor(1.))
    def prepare(self,data):
        self.counts.copy_(torch.bincount(torch.as_tensor(data['y']).long(),minlength=len(self.counts)).to(self.counts))
    def geometry(self):
        factor=torch.tril(self.factor);cov=factor@factor.transpose(-1,-2)+.05*torch.eye(factor.shape[-1],device=factor.device)
        if self.c['variant'] not in {'baseline','no_transport'}:
            relation=-torch.cdist(self.center,self.center).square();head=self.counts>=self.counts.median()
            relation=relation.masked_fill(~head[None],-1e4);relation.fill_diagonal_(-1e4)
            top=relation.topk(min(self.c.get('donors',3),len(head)),dim=-1).indices;mask=torch.zeros_like(relation).scatter(1,top,1)
            coupling=torch.exp((relation-relation.max(-1,keepdim=True).values).clamp_min(-30))*mask
            for _ in range(12):coupling=coupling/coupling.sum(-1,keepdim=True).clamp_min(1e-8);coupling=coupling/coupling.sum(0,keepdim=True).clamp_min(1e-8)
            coupling=coupling/coupling.sum(-1,keepdim=True).clamp_min(1e-8)
            trace=cov.diagonal(dim1=-2,dim2=-1).sum(-1);orientation=cov/trace[:,None,None]
            transported=torch.einsum('cj,jab->cab',coupling,orientation)*trace[:,None,None]
            tail=(self.counts<self.counts.median())[:,None,None];cov=torch.where(tail,.5*cov+.5*transported,cov)
        return cov
    def distances(self,z,cov):
        delta=z[:,None]-self.center;return torch.einsum('nci,cij,ncj->nc',delta,torch.linalg.inv(cov),delta)
    def forward(self,b):
        z=self.encoder(b['x'].float());cov=self.geometry();distance=self.distances(z,cov);radius=F.softplus(self.radius).square()
        score=distance-radius;probes=z*2.5-z.mean(0,keepdim=True)*1.5
        occupancy=torch.sigmoid(-(self.distances(probes,cov)-radius).min(-1).values)
        return {'probability':(-score).softmax(-1),'logits':-score,'unknown_score':score.min(-1).values,'occupancy':occupancy[:,None]}
    def objective(self,b,p):
        labels=b['y'].long();weight=(self.counts.sum()/self.counts.clamp_min(1));loss=F.cross_entropy(p['logits'],labels,weight=weight)
        if self.c['variant'] not in {'baseline','no_conservation'}:
            violation=p['occupancy'].mean()-self.c.get('occupancy_budget',.05)
            loss=loss+self.dual.detach()*violation
            if self.training:self.dual.add_(.01*violation.detach()).clamp_(min=0)
        return loss


def demo(seed,n,c):return classification_demo(seed,n,classes=c.get('classes',3))


def measure(p,d,c):return {'open_probe_occupancy':float(p['occupancy'].mean())}
