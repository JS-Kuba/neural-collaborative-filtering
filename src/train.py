import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator
import wandb
import time

factors_list = [8, 16, 32, 64]

for factors in factors_list:
    try:
        gmf_config = {'alias': f'gmf_factors_{factors}',
                    'num_epoch': 20,
                    'batch_size': 256,
                    # 'optimizer': 'sgd',
                    # 'sgd_lr': 1e-3,
                    # 'sgd_momentum': 0.9,
                    # 'optimizer': 'rmsprop',
                    # 'rmsprop_lr': 1e-3,
                    # 'rmsprop_alpha': 0.99,
                    # 'rmsprop_momentum': 0,
                    'optimizer': 'adam',
                    'adam_lr': 1e-3,
                    'num_users': 6040,
                    'num_items': 3706,
                    'latent_dim': factors,
                    'num_negative': 4,
                    'l2_regularization': 0,  # 0.01
                    'weight_init_gaussian': True,
                    'use_cuda': False,
                    'use_bachify_eval': False,
                    'device_id': 0,
                    'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

        mlp_config = {'alias': f'mlp_factors_{factors}',
                    'num_epoch': 20,
                    'batch_size': 256,  # 1024,
                    'optimizer': 'adam',
                    'adam_lr': 1e-3,
                    'num_users': 6040,
                    'num_items': 3706,
                    'latent_dim': factors,
                    'num_negative': 4,
                    'layers': [2*factors, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
                    'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
                    'weight_init_gaussian': True,
                    'use_cuda': False,
                    'use_bachify_eval': False,
                    'device_id': 0,
                    'pretrain': False,
                    'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
                    'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

        neumf_config = {'alias': f'neumf_factors_{factors}',
                        'num_epoch': 1,
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
                        }

        # Specify the exact model
        # config = gmf_config
        # engine = GMFEngine(config)
        # config = mlp_config
        # engine = MLPEngine(config)
        config = neumf_config
        engine = NeuMFEngine(config)

        logging=True
        if logging:
            notes = ""
            wandb.init(project="ncf-project", config=config, name=f'{config["alias"]}{notes}')

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

