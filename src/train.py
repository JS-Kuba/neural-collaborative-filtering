import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator
import wandb
import time

factors_list = [16]
attention_heads = [2, 4, 8]
layer_norms = [False, True]

for factors in factors_list:
    for a_heads in attention_heads:
        for l_norm in layer_norms:
            try:
                neumf_config = {'alias': f'MHA_neumf_factors_{factors}_attheads_{a_heads}_lnorm_{l_norm}',
                                'num_epoch': 20,
                                'batch_size': 256,
                                'optimizer': 'adam',
                                'adam_lr': 1e-3,
                                'num_users': 6040,
                                'num_items': 3706,
                                'latent_dim_mf': factors,
                                'latent_dim_mlp': factors,
                                'num_negative': 4,
                                # qst layer should be 2x latent_dim_mlp
                                'layers': [2*factors, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
                                'l2_regularization': 0.0000001,
                                'weight_init_gaussian': True,
                                'use_cuda': True,
                                'use_bachify_eval': True,
                                'device_id': 0,
                                'pretrain': False,
                                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
                                'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg4_Epoch100_HR0.5606_NDCG0.2463.model'),
                                'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                                'attention_heads' : a_heads,
                                'l_norm': l_norm,
                                }

                config = neumf_config
                engine = NeuMFEngine(config)

                print(f"Starting training with config: {config['alias']}. Current number of factors: {factors}, A_heads: {a_heads}, L_norm: {l_norm}")

                logging=True
                if logging:
                    wandb.init(project="ncf-project", config=config, name=config["alias"])

                # Load Data
                ml1m_dir = 'data/ml-1m/ratings.dat'
                ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
                # Reindex
                user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
                user_id['userId'] = np.arange(len(user_id))
                ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
                item_id = ml1m_rating[['mid']].drop_duplicates()
                item_id['itemId'] = np.arange(len(item_id))
                ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
                ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
                print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
                print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))
                # DataLoader for training
                sample_generator = SampleGenerator(ratings=ml1m_rating)
                evaluate_data = sample_generator.evaluate_data

                start = time.time()
                for epoch in range(config['num_epoch']):
                    print('Epoch {} starts !'.format(epoch))
                    print('-' * 80)
                    train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
                    loss = engine.train_an_epoch(train_loader, epoch_id=epoch)
                    hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
                    engine.save(config['alias'], epoch, hit_ratio, ndcg)
                    
                    if logging:
                        wandb.log({"loss": loss, "HR@10": hit_ratio, "NDCG@10": ndcg})
                if logging:
                    wandb.log({"time_to_train": time.time() - start})

            finally:
                if logging:
                    wandb.finish()

