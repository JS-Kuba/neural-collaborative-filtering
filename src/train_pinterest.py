import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator
import wandb
import time

factors_list = [32]
attention_heads = [4]
layer_norms = [True]
logging=True

for factors in factors_list:
    for a_heads in attention_heads:
        for l_norm in layer_norms:
            try:
                neumf_config = {'alias': f'MHA_neumf_factors_{factors}_attheads_{a_heads}_lnorm_{l_norm}',
                                'num_epoch': 20,
                                'batch_size': 128,
                                'optimizer': 'adam',
                                'adam_lr': 0.005,
                                'num_users': 55187,
                                'num_items': 9911,
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
                                'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model',
                                'attention_heads' : a_heads,
                                'l_norm': l_norm,
                                }

                config = neumf_config
                engine = NeuMFEngine(config)

                print(f"Starting training with config: {config['alias']}. Current number of factors: {factors}, A_heads: {a_heads}, L_norm: {l_norm}")

                if logging:
                    wandb.init(project="ncf-project", config=config, name=config["alias"]+"_pinterest")

                # Load Data Pinterest
                pin_dir = 'data/pinterest/pinterest-20.train.rating'
                pin_rating = pd.read_csv(pin_dir, sep='\t', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
                # Reindex
                user_id = pin_rating[['uid']].drop_duplicates().reindex()
                user_id['userId'] = np.arange(len(user_id))
                pin_rating = pd.merge(pin_rating, user_id, on=['uid'], how='left')
                item_id = pin_rating[['mid']].drop_duplicates()
                item_id['itemId'] = np.arange(len(item_id))
                pin_rating = pd.merge(pin_rating, item_id, on=['mid'], how='left')
                pin_rating = pin_rating[['userId', 'itemId', 'rating', 'timestamp']]
                print('Range of userId is [{}, {}]'.format(pin_rating.userId.min(), pin_rating.userId.max()))
                print('Range of itemId is [{}, {}]'.format(pin_rating.itemId.min(), pin_rating.itemId.max()))
                sample_generator = SampleGenerator(ratings=pin_rating)
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

