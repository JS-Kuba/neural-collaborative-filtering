import torch
from torch import nn
from gmf import GMF
from mlp import MLP
from engine import Engine
from utils import use_cuda, resume_checkpoint


class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, use_layer_norm):
        super(SelfAttentionLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        attn_output, _ = self.self_attention(x, x, x)  # Self-attention
        return self.layer_norm(x + attn_output)  # Add & Normalize


class NeuMF(torch.nn.Module):
    def __init__(self, config):
        super(NeuMF, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim_mf = config['latent_dim_mf']
        self.latent_dim_mlp = config['latent_dim_mlp']
        self.num_attention_heads = config.get('attention_heads')
        self.use_layer_norm = config.get('l_norm')

        # Embeddings for MLP and MF
        self.embedding_user_mlp = nn.Embedding(self.num_users, self.latent_dim_mlp)
        self.embedding_item_mlp = nn.Embedding(self.num_items, self.latent_dim_mlp)
        self.embedding_user_mf = nn.Embedding(self.num_users, self.latent_dim_mf)
        self.embedding_item_mf = nn.Embedding(self.num_items, self.latent_dim_mf)

        # Self-Attention Layer
        self.self_attention = SelfAttentionLayer(self.latent_dim_mlp * 2, self.num_attention_heads, self.use_layer_norm)

        # Fully Connected Layers for MLP
        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        # Output Layers
        self.affine_output = nn.Linear(config['layers'][-1] + self.latent_dim_mf, 1)
        self.logistic = nn.Sigmoid()

        # Initialize model parameters if specified
        if config['weight_init_gaussian']:
            for sm in self.modules():
                if isinstance(sm, (nn.Embedding, nn.Linear)):
                    torch.nn.init.normal_(sm.weight.data, 0.0, 0.01)

    def forward(self, user_indices, item_indices):
        # User and item embeddings for MLP and MF
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        # Concatenate embeddings for MLP
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # [batch_size, embed_dim*2]

        # Apply self-attention to MLP vector
        mlp_vector = mlp_vector.unsqueeze(1)  # Add sequence dimension: [batch_size, 1, embed_dim*2]
        mlp_vector = self.self_attention(mlp_vector)  # [batch_size, 1, embed_dim*2]
        mlp_vector = mlp_vector.squeeze(1)  # Remove sequence dimension: [batch_size, embed_dim*2]

        # MLP layers
        for layer in self.fc_layers:
            mlp_vector = torch.nn.ReLU()(layer(mlp_vector))

        # Element-wise product for GMF
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        # Concatenate MLP and GMF vectors
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)  # [batch_size, final_dim]

        # Predict rating
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def load_pretrain_weights(self):
        """Load pretrained weights for MLP and GMF models"""
        config = self.config
        config['latent_dim'] = config['latent_dim_mlp']
        mlp_model = MLP(config)
        if config['use_cuda']:
            mlp_model.cuda()
        resume_checkpoint(mlp_model, model_dir=config['pretrain_mlp'], device_id=config['device_id'])

        self.embedding_user_mlp.weight.data = mlp_model.embedding_user.weight.data
        self.embedding_item_mlp.weight.data = mlp_model.embedding_item.weight.data
        for idx in range(len(self.fc_layers)):
            self.fc_layers[idx].weight.data = mlp_model.fc_layers[idx].weight.data

        config['latent_dim'] = config['latent_dim_mf']
        gmf_model = GMF(config)
        if config['use_cuda']:
            gmf_model.cuda()
        resume_checkpoint(gmf_model, model_dir=config['pretrain_mf'], device_id=config['device_id'])

        self.embedding_user_mf.weight.data = gmf_model.embedding_user.weight.data
        self.embedding_item_mf.weight.data = gmf_model.embedding_item.weight.data

        self.affine_output.weight.data = 0.5 * torch.cat([mlp_model.affine_output.weight.data,
                                                          gmf_model.affine_output.weight.data], dim=-1)
        self.affine_output.bias.data = 0.5 * (mlp_model.affine_output.bias.data + gmf_model.affine_output.bias.data)


class NeuMFEngine(Engine):
    """Engine for training & evaluating NeuMF model"""
    def __init__(self, config):
        self.model = NeuMF(config)
        if config['use_cuda']:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(NeuMFEngine, self).__init__(config)
        print(self.model)

        if config['pretrain']:
            self.model.load_pretrain_weights()
