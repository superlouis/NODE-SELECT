from .baseline import *

class SIMPLE_LAYER(MessagePassing):
    def __init__(self, feat_in, feat_out,spread=1,bias=False):
        super(SIMPLE_LAYER, self).__init__(aggr='mean',flow='target_to_source')
        self.L_flag       = torch.zeros(1,1).cuda()
        self.bias         = bias

        self.updater      = Linear(feat_in,feat_out,bias=self.bias)
        self.p_leader     = Linear(feat_out, 2,bias=self.bias)
        self.layer_weight = Linear(2*feat_out,1,bias=self.bias)
        self.aggregator   = PROPAGATION_OUT()

    
    def forward(self,x,edge_index, T):
        # PRELIMINARY CALCULATIONS
        self.L_flag  = torch.zeros(1,1).cuda()

        depth        = 0
        self.L_flag  = self.L_flag*0
        updated_x    = F.leaky_relu(self.updater(x))
        sum_Neigh_x  = self.aggregator(updated_x, edge_index)

        #  SELECTION  <==============================================================
        random_prob  = F.relu(self.p_leader(sum_Neigh_x)) 
        random_prob  = F.softmax(random_prob,dim=-1)

        self.prob_i  = random_prob[:,1].unsqueeze(1)
        hot_prob     =  torch.where(random_prob[:,1]>T,torch.tensor(1).cuda(),torch.tensor(0).cuda())
        SEL_v        = hot_prob.view(-1,1)
        self.L_flag  = self.L_flag.float().view(-1,1) + SEL_v

        #  SUMMATION + CONCAT <========================================================
        sum_SEL_x    = self.aggregator(SEL_v*updated_x,edge_index)
        concat_sums  = torch.cat([sum_SEL_x,sum_Neigh_x],dim=-1)

        #  WEIGHT             <========================================================
        weight_SEL_v = torch.sigmoid(self.layer_weight(concat_sums))
        A_x          = F.relu(self.aggregator(weight_SEL_v* SEL_v* updated_x,edge_index))

        out          = updated_x + A_x
        return out

class COMPLEX_LAYER(MessagePassing):
    def __init__(self, feat_in, feat_out, spread=2, bias=False):
        super(COMPLEX_LAYER, self).__init__(aggr='mean',flow='target_to_source')
        self.spread       = spread
        self.bias         = bias

        self.L_flag       = torch.zeros(1,1).cuda()
        self.onehot_layer = torch.eye(spread).cuda()

        self.updater      = Linear(feat_in,feat_out,bias=self.bias)
        self.p_leader     = Linear(feat_out, 2,bias=self.bias)
        self.layer_weight = Linear(spread+feat_out,1, bias=self.bias)
        self.aggr_weight  = Linear(feat_out*2,1, bias=self.bias)
        self.alpha        = None
        
        self.aggregator   = PROPAGATION_OUT()

    def find_next_lead(self,src_ct, dst, prev_leader,coverage_tensor):
        next_lead            = torch.zeros_like(prev_leader).cuda()
        expand_src_id        = prev_leader.repeat_interleave(src_ct)
        coverage_tensor      = coverage_tensor + expand_src_id.float()
        dst_nodes            = torch.unique(dst*expand_src_id)
        next_lead[dst_nodes] = 1
        return next_lead, coverage_tensor

    def forward(self,x,edge_index,T):
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

        x            = F.leaky_relu(self.updater(x))
        updated_x    = x
        geo_x        = updated_x[:]
        local_x      = self.aggregator(updated_x, edge_index)

        random_prob  = torch.sigmoid(self.p_leader(updated_x))
        random_prob  = F.softmax(random_prob,dim=-1)

        self.prob_i  = random_prob[:,1]

        hot_prob     = torch.where(random_prob>T,torch.tensor(1).cuda(),torch.tensor(0).cuda())
        kept_prob    = torch.where(random_prob>T,random_prob,torch.tensor(0.0).cuda())
        
        leader_hot   = hot_prob[:,1].unsqueeze(dim=1)
        leader_prob  = kept_prob[:,1].unsqueeze(dim=1)

        self.L_flag  = self.L_flag.float() + leader_hot.float()
        while layer  < self.spread:
            layer_i  = self.onehot_layer[layer].float().unsqueeze(0)
            layer_i  = layer_i.repeat_interleave(geo_x.size(0),dim=0)
            layer_i  = torch.cat([layer_i,local_x],dim=-1)

            layer_i  = layer_i * leader_hot.float()
            weight_l = torch.sigmoid(self.layer_weight(layer_i))
            geopass  = weight_l*geo_x

            temp_agg      = self.aggregator(geopass*leader_hot.float(),edge_index)
            geo_x         = geo_x + temp_agg
            leader_hot, _ = self.find_next_lead(src_counts, dst, leader_hot,coverage)
            leader_prob   = kept_prob[:,1].unsqueeze(dim=1)*leader_hot
            layer        += 1

        some_X       = torch.cat([geo_x,updated_x],dim=-1)
        some_weight  = torch.sigmoid(self.aggr_weight(some_X))
        out          = (1-some_weight)*updated_x+(some_weight*geo_x)
        self.alpha   = some_weight

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

        self.W_layers      = nn.ModuleList([SIMPLE_LAYER(feat_in,feat_out) for i in range(learners)])
        
        # *** N.B. uncomment line below (143 ) when using smaller datasets
        # self.W_layers      = nn.ModuleList([COMPLEX_LAYER(feat_in,feat_out,prob_to_lead=self.p2L, spread=self.spread) for i in range(learners)])
        
        self.fusing_weight = unifying_weight
        if unifying_weight : self.prob_weighter = Linear(self.N_learners,feat_out, bias=True)
        self.leader_info   = None
        self.L_containers  = None

    def forward(self,data):
        # INITIALIZATION 
        x, edge_index        = data.x,  data.edge_index
        L_containers         = [None]*self.N_learners
        filter_output        = [None]*self.N_learners
        filter_vstar         = np.zeros(self.N_learners)
        X_out                = torch.zeros(x.size(0),self.dim_out).cuda()

        # CONVOLUTION-PER-LEARNER
        for i in range(self.N_learners):
            x_temp           = self.W_layers[i](x,edge_index,self.p2L)
            x_temp           = F.leaky_relu(x_temp,self.neg_slope)
            x_temp           = F.dropout(x_temp,p=0.4, training=self.training)
            X_out            = X_out + x_temp
            filter_output[i] = F.softmax(x_temp.unsqueeze(0),dim=-1)

            filter_vstar[i]  = (self.W_layers[i].L_flag.sum()/x.size(0)).item()
            L_containers[i]  = self.W_layers[i].L_flag


        # RE-ARRANGING INFORMATION
        self.outputs       = torch.cat(filter_output,dim=0)
        self.leader_info   = filter_vstar#np.round_(filter_vstar,2)
        L_containers       = torch.cat(L_containers,dim=-1)
        self.L_containers  = L_containers.sum(dim=-1).float()/self.N_learners

        # FEATURE-SUMMATION 
        if self.fusing_weight:
            global_weight  = self.prob_weighter(L_containers)
            global_weight  = F.relu(global_weight)
        else:
            global_weight  = 0.0

        X_out              = X_out + global_weight 

        # FEATURE-SUMMATION 
        y                  = X_out        
        y                  = F.log_softmax(y,dim=-1)
        y                  = y.squeeze()
        return y

      