from .baseline import *

class WAVE_LAYER(MessagePassing):
    def __init__(self, feat_in, feat_out, spread=2, prob_to_lead = 0.4):
        super(WAVE_LAYER, self).__init__(aggr='mean',flow='target_to_source')
        self.spread       = spread
        self.threshold    = prob_to_lead

        self.L_flag       = torch.zeros(1,1).cuda()
        self.onehot_layer = torch.eye(spread).cuda()

        self.updater      = Linear(feat_in,feat_out,bias=False)
        self.p_leader     = Linear(feat_out, 2, bias=False)
        self.layer_weight = Linear(spread+feat_out,1, bias=False)
        self.aggr_weight  = Linear(feat_out*2,1, bias=False)
        
        self.aggregator   = PROPAGATION_OUT()

    def find_next_lead(self,src_ct, dst, prev_leader,coverage_tensor):
        next_lead            = torch.zeros_like(prev_leader).cuda()
        expand_src_id        = prev_leader.repeat_interleave(src_ct)
        coverage_tensor      = coverage_tensor + expand_src_id.float()
        dst_nodes            = torch.unique(dst*expand_src_id)
        next_lead[dst_nodes] = 1
        return next_lead, coverage_tensor

    def forward(self,x,edge_index):
        # INITIALIZATION
        layer        = 0
        coverage     = torch.zeros(1).cuda()
        src, dst     = edge_index
        self.L_flag  = self.L_flag*0
        self.weights_1    = torch.tensor([]).cuda()

        # PRELIMINARY CALCULATIONS
        R,src_counts = torch.unique(src,return_counts=True)
        if src_counts.size(0) != x.size(0):
            temp_ct    = torch.zeros(x.size(0),dtype=torch.long).cuda()
            temp_ct[R] = src_counts
            src_counts = temp_ct

        x          = F.leaky_relu(self.updater(x))
        updated_x    = x
        geo_x        = updated_x[:]
        local_x      = self.aggregator(updated_x, edge_index)
        random_prob  = torch.sigmoid(self.p_leader(local_x))

        random_prob  = F.softmax(random_prob,dim=-1)
        hot_prob     = torch.where(random_prob>self.threshold,torch.tensor(1).cuda(),torch.tensor(0).cuda())
        leader_hot   = hot_prob[:,1].unsqueeze(dim=1)
        self.L_flag  = self.L_flag.float() + leader_hot.float()

        while layer  < self.spread:
            layer_i  = self.onehot_layer[0].float().unsqueeze(0)
            layer_i  = layer_i.repeat_interleave(geo_x.size(0),dim=0)
            layer_i  = torch.cat([layer_i,geo_x],dim=-1)
            weight_l = torch.sigmoid(self.layer_weight(layer_i))
            geopass  = weight_l*geo_x

            temp_agg      = self.aggregator(geopass*leader_hot.float(),edge_index)
            geo_x         = geo_x + temp_agg
            leader_hot, _ = self.find_next_lead(src_counts, dst, leader_hot,coverage)
            layer        += 1

        some_X       = torch.cat([geo_x,updated_x],dim=-1)
        some_weight  = torch.sigmoid(self.aggr_weight(some_X))
        out          = (1-some_weight)*updated_x+(some_weight*geo_x)
        return out

class PROPAGATION_OUT(MessagePassing):
    def __init__(self):
        super(PROPAGATION_OUT, self).__init__(aggr='add', flow="target_to_source")

    def forward(self, x, edge_index) : return self.propagate(edge_index,x=x)
    def message(self,x_j)            : return x_j
    def update(self,aggr_out)        : return aggr_out

class NSGNN(torch.nn.Module):
    def __init__(self, feat_in, feat_out, learners = 1, p2L=0.48, neg_slope=0.0, spread_L=1, unifying_weight=True):
        super(NSGNN, self).__init__()
        self.N_learners    = learners
        self.dim_out       = feat_out
        self.p2L           = p2L
        self.spread        = spread_L
        self.neg_slope     = neg_slope
        self.W_layers      = nn.ModuleList([WAVE_LAYER(feat_in,feat_out,prob_to_lead=self.p2L, spread=self.spread) for i in range(learners)])
        
        self.fusing_weight = unifying_weight
        if unifying_weight : self.prob_weighter = Linear(self.N_learners,feat_out, bias=False)
        self.leader_info   = None

    def forward(self,data):
        # INITIALIZATION
        x, edge_index        = data.x,  data.edge_index
        L_containers         = torch.tensor([]).cuda()

        X_out                = torch.zeros(x.size(0),self.dim_out).cuda()
        adj_prob             = torch.randn(x.size(0),2).cuda()
        agreement_tensor     = torch.zeros(x.size(0),1).cuda()
        lead_pct_tensor      = []

        temp_thing           = torch.tensor([]).cuda()


        # CONVOLUTION-PER-LEARNER
        for i in range(self.N_learners):
            x_temp           = self.W_layers[i](x,edge_index)
            x_temp           = F.leaky_relu(x_temp,self.neg_slope)
            x_temp           = F.dropout(x_temp,p=0.4, training=self.training)
            X_out            = X_out + x_temp
            temp_thing       = torch.cat([temp_thing,F.softmax(x_temp.unsqueeze(0),dim=-1)],dim=0)
            L_containers     = torch.cat([L_containers,self.W_layers[i].L_flag.unsqueeze(0)],dim=0)
            lead_pct_tensor.append(self.W_layers[i].L_flag.sum()/x.size(0))

        self.outputs   = temp_thing
        # FEATURE-SUMMATION 
        if self.fusing_weight:
            P_out          = torch.transpose(L_containers,0,1).squeeze(-1)
            global_weight  = self.prob_weighter(P_out)
            global_weight  = torch.sigmoid(global_weight)
        else:
            global_weight  = 0.0

        # FEATURE-SUMMATION 
        y                  = X_out+global_weight        
        y                  = F.log_softmax(y,dim=-1)
        y                  = y.squeeze()

        lead_pct_tensor    = torch.tensor(lead_pct_tensor)
        self.leader_info   = np.round_(lead_pct_tensor.cpu().tolist(),2)
        return y

