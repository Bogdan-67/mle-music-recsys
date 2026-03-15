#!/usr/bin/env python
# coding: utf-8

# # Инициализация

# Загружаем библиотеки необходимые для выполнения кода ноутбука.

# In[2]:


import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.sparse import csr_matrix, coo_matrix
import polars as pl
import gc


# # === ЭТАП 1 ===

# # Загрузка первичных данных

# Загружаем первичные данные из файлов:
# - tracks.parquet
# - catalog_names.parquet
# - interactions.parquet

# In[2]:


tracks = pd.read_parquet("tracks.parquet")
tracks.head()


# In[3]:


catalog_names = pd.read_parquet('catalog_names.parquet')
catalog_names.head()


# In[2]:


import polars as pl

int_pl = pl.scan_parquet('interactions.parquet')
interactions = int_pl.collect().drop('__index_level_0__').to_pandas()


# # Обзор данных

# Проверяем данные, есть ли с ними явные проблемы.

# In[5]:


tracks.info()


# In[ ]:


print(tracks['albums'].isna().sum())
print(tracks['artists'].isna().sum())
print(tracks['genres'].isna().sum())


# In[ ]:


print(f"Количество треков без альбомов: {(tracks['albums'].str.len() == 0).sum()}")
print(f"Количество треков без артистов: {(tracks['artists'].str.len() == 0).sum()}")
print(f"Количество треков без жанров: {(tracks['genres'].str.len() == 0).sum()}")


# Отфильтруем треки, у которых нет жанров, потому что в построенной матрице они все равно будут иметь пустые строки

# In[6]:


tracks = tracks[tracks['genres'].str.len() > 0]


# In[ ]:


tracks['track_id'] = tracks['track_id'].astype('int32')


# In[ ]:


catalog_names.info()


# In[ ]:


catalog_names['type'].unique()


# In[ ]:


interactions.info()


# In[ ]:


interactions['track_seq'].isna().any()


# In[3]:


print(f"Количество уникальных пользователей: {interactions['user_id'].nunique()}")


# In[7]:


def optimize_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=['int', 'float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer' if 'int' in str(df[col].dtype) else 'float')

    return df


# In[8]:


tracks = optimize_numeric_types(tracks)
interactions = optimize_numeric_types(interactions)


# In[9]:


tracks.info()


# In[12]:


interactions.info()


# # Выводы

# Приведём выводы по первому знакомству с данными:
# - есть ли с данными явные проблемы,
# - какие корректирующие действия (в целом) были предприняты.

# В исходном наборе данных имеются пропуски среди альбомов, артистов и жанров. Было принято решение отбросить все треки, у которых не заполено поле с жанрами, поскольку именно этот признак будет использоваться для построения матрицы взаимодействий. Остальные признаки не должны сильно повлиять на качество рекомендаций

# # === ЭТАП 2 ===

# # EDA

# Распределение количества прослушанных треков.

# In[13]:


listened_cnt = interactions.groupby('user_id')['track_id'].count().reset_index()

sns.histplot(
    listened_cnt['track_id'], 
    log_scale=True,
    kde=True)


# Наиболее популярные треки

# In[14]:


track_listenings = interactions['track_id'].value_counts().reset_index()
track_listenings.columns= ['track_id', 'listenings']
most_popular = track_listenings.sort_values(by='listenings', ascending=False).head(10)
most_popular = most_popular.merge(catalog_names[catalog_names['type'] == 'track'][['id', 'name']], left_on='track_id', right_on='id', how='left').drop(columns=['id'])

most_popular


# Наиболее популярные жанры

# In[15]:


most_popular_genres = track_listenings.merge(tracks[['track_id', 'genres']], on='track_id', how='left')\
    .explode(column=['genres'], ignore_index=True)\
    .groupby('genres')['listenings'].sum().reset_index()\
    .sort_values(by='listenings', ascending=False)\
    .merge(catalog_names[catalog_names['type'] == 'genre'][['id', 'name']], left_on='genres', right_on='id', how='left')\
    .drop(columns=['genres']).head(10)

most_popular_genres


# Треки, которые никто не прослушал

# In[ ]:


track_listenings.query("listenings == 0")


# In[ ]:


track_listenings.sort_values(by='listenings', ascending=False).tail()


# In[ ]:


track_ids = tracks['track_id'].unique().tolist()

interactions[~interactions['track_id'].isin(track_ids)]['track_id'].count()


# # Преобразование данных

# Преобразуем данные в формат, более пригодный для дальнейшего использования в расчётах рекомендаций.

# In[16]:


items = tracks.copy()
events = interactions.copy()


# In[17]:


items = items.rename(columns={'track_id': 'item_id'})
events = events.rename(columns={'track_id': 'item_id'})


# # Сохранение данных

# Сохраним данные в двух файлах в персональном S3-бакете по пути `recsys/data/`:
# - `items.parquet` — все данные о музыкальных треках,
# - `events.parquet` — все данные о взаимодействиях.

# In[19]:


items.to_parquet("items.parquet", index=False)
events.to_parquet("events.parquet", index=False)


# In[27]:


import boto3
import os
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client('s3', endpoint_url=os.environ['S3_ENDPOINT_URL'])

s3.upload_file('items.parquet', os.environ['S3_BUCKET_NAME'], 'recsys/data/items.parquet')
s3.upload_file('events.parquet', os.environ['S3_BUCKET_NAME'], 'recsys/data/events.parquet')


# # Очистка памяти

# Здесь, может понадобится очистка памяти для высвобождения ресурсов для выполнения кода ниже. 
# 
# Приведите соответствующие код, комментарии, например:
# - код для удаление более ненужных переменных,
# - комментарий, что следует перезапустить kernel, выполнить такие-то начальные секции и продолжить с этапа 3.

# In[28]:


import gc

del tracks
del interactions
del track_listenings
del most_popular
del most_popular_genres
del listened_cnt

gc.collect()


# # === ЭТАП 3 ===

# # Загрузка данных

# Если необходимо, то загружаем items.parquet, events.parquet.

# In[2]:


items = pd.read_parquet('items.parquet')


# In[1]:


import polars as pl

events_pl = pl.scan_parquet('events.parquet')

events = events_pl.collect().to_pandas()


# In[4]:


events.head()


# # Разбиение данных

# Разбиваем данные на тренировочную, тестовую выборки.

# In[6]:


item_ids = items['item_id'].unique().tolist()

events_clean = events[events['item_id'].isin(item_ids)]


# In[7]:


global_time_split_date = pd.to_datetime('2022-12-17')

global_time_split_idx = events_clean['started_at'] < global_time_split_date
events_train = events_clean[global_time_split_idx]
events_test = events_clean[~global_time_split_idx]


# In[8]:


del item_ids
del global_time_split_idx
gc.collect()


# # Топ популярных

# Рассчитаем рекомендации как топ популярных.

# In[33]:


top_popular = events['item_id'].value_counts().reset_index()
top_popular.columns = ['item_id', 'listenings']

top_popular = top_popular.sort_values(by='listenings', ascending=False).head(100)

top_popular.to_parquet('top_popular.parquet')


# In[ ]:


top_popular.head()


# # Персональные

# Рассчитаем персональные рекомендации.

# In[7]:


import sklearn.preprocessing

user_encoder = sklearn.preprocessing.LabelEncoder()
user_encoder.fit(events["user_id"])
events_train["user_id_enc"] = user_encoder.transform(events_train["user_id"])
events_test["user_id_enc"] = user_encoder.transform(events_test["user_id"])

item_encoder = sklearn.preprocessing.LabelEncoder()
item_encoder.fit(items["item_id"])
items["item_id_enc"] = item_encoder.transform(items["item_id"])
events_train["item_id_enc"] = item_encoder.transform(events_train['item_id'])
events_test["item_id_enc"] = item_encoder.transform(events_test['item_id'])


# In[8]:


del events
gc.collect()


# In[ ]:


import joblib

joblib.dump(user_encoder, 'user_encoder.pkl')
joblib.dump(item_encoder, 'item_encoder.pkl')


# In[2]:


import joblib

with open('user_encoder.pkl', 'rb') as f:
    user_encoder = joblib.load(f)

with open('item_encoder.pkl', 'rb') as f:
    item_encoder = joblib.load(f)


# In[ ]:


events_train['item_id_enc'].max()


# In[7]:


len(events_train)


# Ограничиваю тестовую выборку до 100 последних действий для оптимизации использования памяти

# In[10]:


min_interactions = 5
max_events_per_user = 100

user_counts = events_train['user_id'].value_counts()
active_users = user_counts[user_counts >= min_interactions].index
del user_counts
gc.collect()

events_train = events_train[events_train['user_id'].isin(active_users)]
del active_users
gc.collect()

print(f"Размер с минимальным количеством взаимодействий {min_interactions}: {len(events_train)}")

events_train.sort_values(by='started_at', ascending=False, inplace=True)
rank = events_train.groupby('user_id').cumcount()

events_train = events_train[rank < max_events_per_user].reset_index(drop=True)
del rank
gc.collect()

print(f"Размер с последними {max_events_per_user} взаимодействиями: {len(events_train)}")
events_train.info(memory_usage='deep')


# In[13]:


events_train.to_parquet('events_train.parquet')


# In[ ]:


events_test.to_parquet('events_test.parquet')


# In[3]:


events_train = pd.read_parquet('events_train.parquet')


# In[ ]:


u = events_train['user_id_enc'].values
i = events_train['item_id_enc'].values

data = np.ones(len(events_train), dtype='int32')

num_users = len(user_encoder.classes_)
num_items = len(item_encoder.classes_)

matrix = coo_matrix((data, (u, i)), shape=(num_users, num_items))

matrix = matrix.astype(np.float32)

user_item_matrix_train = matrix.tocsr()

del data
del u
del i
del matrix
gc.collect()


# In[5]:


joblib.dump(user_item_matrix_train, 'user_item_matrix_train.pkl')


# In[3]:


with open('user_item_matrix_train.pkl', 'rb') as f:
    user_item_matrix_train = joblib.load(f)


# In[ ]:


import gc

del events_clean
del events_train
del events_test

gc.collect()


# In[11]:


def get_sparse_matrix_mem(matrix):
    mem = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    print(f"Вес CSR матрицы: {mem / 1024**2:.2f} MB")
    print(f"Вес CSR матрицы: {mem / 1024**3:.2f} GB")
    
get_sparse_matrix_mem(user_item_matrix_train)


# In[4]:


from implicit.als import AlternatingLeastSquares

als_collab_model = AlternatingLeastSquares(factors=32, iterations=10, regularization=0.05)
als_collab_model.fit(user_item_matrix_train)


# In[5]:


als_collab_model.save('als_model.npz')


# In[6]:


import numpy as np

np.save('user_factors.npy', als_collab_model.user_factors)
np.save('item_factors.npy', als_collab_model.item_factors)


# In[7]:


import psutil
import os

process = psutil.Process(os.getpid())
print(f"Общее потребление процессом: {process.memory_info().rss / 1024**3:.2f} GB")


# In[9]:


def get_model_memory_usage(model):
    user_factors = model.user_factors
    item_factors = model.item_factors
    
    user_mem = user_factors.nbytes
    item_mem = item_factors.nbytes
    total_mem = user_mem + item_mem
    
    print(f"User Factors: {user_mem / 1024**2:.2f} MB")
    print(f"Item Factors: {item_mem / 1024**2:.2f} MB")
    print(f"---")
    print(f"Общий вес модели в RAM: {total_mem / 1024**2:.2f} MB ({total_mem / 1024**3:.2f} GB)")

get_model_memory_usage(als_collab_model)


# In[1]:


import numpy as np
from implicit.als import AlternatingLeastSquares

data = np.load('als_model.npz', allow_pickle=True)

factors = data['user_factors'].shape[1]
als_collab_model = AlternatingLeastSquares(factors=factors)

als_collab_model.user_factors = data['user_factors']
als_collab_model.item_factors = data['item_factors']

del data
del factors


# In[25]:


import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

def recommend_with_batches(model, save_filename, user_item_matrix, user_encoder, item_encoder, schema, batch_size=5000, N_rec=30, filter_already_liked_items=True):
    user_ids_encoded = np.arange(len(user_encoder.classes_))

    n_batches = int(np.ceil(len(user_ids_encoded) / batch_size))

    print(f"Total batches: {n_batches}")

    if schema is None:
        schema = pa.schema([
            ('user_id', pa.int32()),
            ('item_id', pa.int32()),
            ('score', pa.float32())
        ])

    writer = pq.ParquetWriter(save_filename, schema=schema)

    try:
        for i in tqdm(range(n_batches)):
            print(f"Procesing batch {i}/{n_batches}")
            start = i * batch_size
            end = min(len(user_ids_encoded), (i+1)*batch_size)
            batch_items = user_ids_encoded[start:end]

            ids, scores = model.recommend(
                batch_items,
                user_item_matrix[batch_items],
                filter_already_liked_items=filter_already_liked_items,
                N=N_rec
            )

            real_user_ids = user_encoder.inverse_transform(batch_items)
            user_column = np.repeat(real_user_ids, N_rec)

            ids_flat = ids.reshape(-1)
            scores_flat = scores.reshape(-1)

            valid_mask = ids_flat != -1

            valid_users = user_column[valid_mask]
            valid_items = item_encoder.inverse_transform(ids_flat[valid_mask])
            valid_scores = scores_flat[valid_mask]

            batch_df = pd.DataFrame({
                'user_id': valid_users,
                'item_id': valid_items,
                'score': valid_scores
            })

            table = pa.Table.from_pandas(batch_df, schema=schema)
            writer.write_table(table)

            del batch_df, batch_items, ids, scores, real_user_ids, user_column,
            ids_flat, scores_flat, valid_users, valid_items, valid_scores
            gc.collect()

    except Exception as e:
        print(f"Error: {e}")
        writer.close()
        raise

    finally:
        writer.close()


# In[ ]:


recommend_with_batches(
    model=als_collab_model,
    save_filename='personal_als.parquet', 
    user_item_matrix=user_item_matrix_train, 
    user_encoder=user_encoder, 
    item_encoder=item_encoder,
    batch_size=10000
)


# # Похожие

# Рассчитаем похожие, они позже пригодятся для онлайн-рекомендаций.

# In[8]:


import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from tqdm.auto import tqdm

filename = 'similar_items.parquet'
item_ids_enc = np.arange(len(item_encoder.classes_))
batch_size = 1000
N_sim = 20

n_batches = int(np.ceil(len(item_ids_enc)/batch_size))

schema = pa.schema([
    ('item_id', pa.int32()),
    ('similar_item_id', pa.int32()),
    ('score', pa.float32())
])

writer = pq.ParquetWriter(filename, schema, compression='snappy')

try:
    for i in tqdm(range(n_batches)):
        print(f"Batch {i+1}/{n_batches}")

        start = i * batch_size
        end = min((i+1)*batch_size, len(item_ids_enc))
        batch_items = item_ids_enc[start:end]
        
        ids, scores = als_collab_model.similar_items(
            batch_items,
            N=N_sim
        )

        real_item_ids = item_encoder.inverse_transform(batch_items)
        item_column = np.repeat(real_item_ids, N_sim)

        sim_item_column = item_encoder.inverse_transform(ids.reshape(-1))
        score_column = scores.reshape(-1)

        batch_df = pd.DataFrame({
            'item_id': item_column,
            'similar_item_id': sim_item_column,
            'score': score_column
        })

        table = pa.Table.from_pandas(batch_df, schema=schema)
        writer.write_table(table)

finally:
    writer.close()


# In[9]:


similar_items = pd.read_parquet('similar_items.parquet')


# In[10]:


len(similar_items)


# In[4]:


items = pd.read_parquet('items.parquet')


# In[4]:


def get_item_genre_matrix(items, genres, item_encoder, genres_encoder):
    items['item_id_enc'] = item_encoder.transform(items['item_id'])
    exploded_items = items.explode('genres')

    exploded_items['genre_id_enc'] = genres_encoder.transform(exploded_items['genres'])

    item_indeces = exploded_items['item_id_enc'].values
    genre_indeces = exploded_items['genre_id_enc'].values

    num_rows = len(items)
    num_cols = len(genres)

    data = np.ones(len(exploded_items), dtype='int32')

    item_genre_matrix = csr_matrix((data, (item_indeces, genre_indeces)), shape=(num_rows, num_cols))

    del exploded_items, item_indeces, genre_indeces, data
    gc.collect()

    return item_genre_matrix


# In[5]:


catalog_names = pd.read_parquet('catalog_names.parquet')


# In[8]:


genres = items.explode('genres')['genres'].unique()


# In[9]:


from sklearn.preprocessing import LabelEncoder

genres_encoder = LabelEncoder()
genres_encoded = genres_encoder.fit_transform(genres)

joblib.dump(genres_encoder, 'genres_encoder.pkl')


# In[10]:


item_genre_matrix = get_item_genre_matrix(items, genres_encoded, item_encoder, genres_encoder)


# In[11]:


from implicit.nearest_neighbours import BM25Recommender

# 1. Транспонируем матрицу, чтобы треки стали КОЛОНКАМИ (форма станет: Жанры х Треки)
# Обязательно делаем .tocsr(), так как implicit ожидает именно CSR формат!
genre_item_matrix = item_genre_matrix.T.tocsr()

# 2. Обучаем модель
content_model = BM25Recommender(K=100)
content_model.fit(genre_item_matrix)

# ДЛЯ ПРОВЕРКИ: должно вывести (1000000, 1000000), а не (166, 166)
print(f"Размер матрицы сходства треков: {content_model.similarity.shape}")


# In[12]:


content_model.save('content_model.npz')


# In[ ]:


recommend_with_batches(
    model=content_model,
    save_filename='content_recs.parquet', 
    user_item_matrix=user_item_matrix_train, 
    user_encoder=user_encoder,
    item_encoder=item_encoder,
    batch_size=5000
)


# # Построение признаков

# Построим три признака, можно больше, для ранжирующей модели.

# In[ ]:


def get_item_features(events):
    # Количество прослушиваний за последнюю неделю
    last_date = events['started_at'].max()
    start_date = last_date - pd.Timedelta(days=7)

    print("Подсчет количества прослушиваний за последнюю неделю")
    weekly_stats = events[['item_id', 'started_at']]\
        [events['started_at'] >= start_date]['item_id']\
        .value_counts().reset_index(name='weekly_listenings')

    # Количество прослушиваний за последний месяц
    start_date = last_date - pd.Timedelta(days=30)

    print("Подсчет количества прослушиваний за последний месяц")
    monthly_stats = events[['item_id', 'started_at']]\
        [events['started_at'] >= start_date]['item_id']\
        .value_counts().reset_index(name='monthly_listenings')

    item_features = pd.merge(weekly_stats, monthly_stats, on='item_id', how='outer').fillna(0)

    # Скользящее среднее прослушиваний за последние 7 дней
    print("Подсчет скользящего среднего прослушиваний за последние 7 дней")
    daily_stats = events.groupby(['started_at', 'item_id']).size().reset_index(name='daily_plays')
    daily_stats = daily_stats.sort_values(['item_id', 'started_at'])
    group = daily_stats.groupby('item_id')['daily_plays']
    daily_stats['moving_avg_7d'] = (group.transform(lambda x: x.rolling(window=7, min_periods=1).mean()).astype('float32'))

    # Относительный прирост (скорость)
    print("Подсчет относительного прироста (скорости)")
    daily_stats['velocity'] = (
        daily_stats['daily_plays'] / (daily_stats['moving_avg_7d'] + 1e-5)
    ).astype('float32')

    # Линейный тренд
    print("Подсчет линейного тренда")
    daily_stats['absolute_growth'] = group.diff(periods=7).fillna(0)

    # Экспоненциальное среднее (EWMA)
    print("Подсчет экспоненциального среднего")
    # Быстрое среднее
    daily_stats['ema_fast'] = group.transform(lambda x: x.ewm(span=3).mean())
    # Медленное среднее
    daily_stats['ema_slow'] = group.transform(lambda x: x.ewm(span=7).mean())

    # Признак импульса
    print("Подсчет импульса")
    daily_stats['momentum'] = (daily_stats['ema_fast'] / (daily_stats['ema_slow'] + 1e-5)).astype('float32')


    latest_stats = daily_stats.groupby('item_id').last().reset_index()
    latest_stats = latest_stats[['item_id', 'moving_avg_7d', 'velocity', 'absolute_growth', 'momentum']]

    item_features = item_features.merge(latest_stats, on='item_id', how='left').fillna(0)

    return item_features


# In[16]:


from tqdm.auto import tqdm 
import gc 

def get_user_top_genres_features(events, items, N=3, chunk_size=200000):
    listen_stats = events.groupby(['user_id', 'item_id']).size().reset_index(name='listen_count')

    total_genres = items['genres'].explode().dropna().nunique()

    item_genres = items[['item_id', 'genres']].dropna(subset=['genres']).copy()

    unique_users = listen_stats['user_id'].unique()
    num_chunks = int(np.ceil(len(unique_users) / chunk_size))

    results = []

    print(f"Обработка жанров по батчам ({num_chunks} батчей)...")

    for i in tqdm(range(num_chunks)):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(unique_users))
        chunk_users = unique_users[start:end]

        chunk_listen_stats = listen_stats[listen_stats['user_id'].isin(chunk_users)]

        merged = chunk_listen_stats.merge(item_genres, on='item_id', how='left')

        del chunk_listen_stats
        gc.collect()

        exploded = merged.explode('genres')
        exploded = exploded[exploded['genres'].astype(str) != '']

        del merged
        gc.collect()

        user_genre_counts = exploded.groupby(['user_id', 'genres'])['listen_count'].sum().reset_index()

        del exploded
        gc.collect()

        # Прослушанные жанры пользователя к общему количеству жанров
        user_genre_ratio = user_genre_counts.groupby('user_id')['genres'].nunique().reset_index(name='genres_count')
        user_genre_ratio['genre_ratio'] = (user_genre_ratio['genres_count'] / total_genres).astype(np.float32)
        user_genre_ratio.drop(columns=['genres_count'], inplace=True)

        user_genre_counts = user_genre_counts.sort_values(by=['user_id', 'listen_count'], ascending=[True, False])

        top_N_genres_per_user = user_genre_counts.groupby('user_id').head(N).copy()

        del user_genre_counts
        gc.collect()

        top_N_genres_per_user['rank'] = top_N_genres_per_user.groupby('user_id').cumcount() + 1

        user_top_genres = top_N_genres_per_user.pivot(index='user_id', columns='rank', values='genres').reset_index()

        del top_N_genres_per_user
        gc.collect()

        col_names = ['user_id'] + [f'top_genre_{i}' for i in user_top_genres.columns[1:]]
        user_top_genres.columns = col_names

        user_top_genres = user_top_genres.merge(user_genre_ratio, on='user_id', how='left')

        results.append(user_top_genres)
        del user_genre_ratio, user_top_genres 
        gc.collect()

    print("Склейка результатов...")
    final_df = pd.concat(results, ignore_index=True)
    
    del listen_stats, item_genres, results
    gc.collect()
    
    return final_df

def get_user_unique_artists(events, items, chunk_size=200000):
    dedup = events[['user_id', 'item_id']].drop_duplicates()

    item_artists = items[['item_id', 'artists']].dropna(subset=['artists']).copy()

    unique_users = dedup['user_id'].unique()
    num_chunks = int(np.ceil(len(unique_users) / chunk_size))

    results = []

    print(f"Обработка артистов по батчам ({num_chunks} батчей)...")

    for i in tqdm(range(num_chunks)):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(unique_users))
        chunk_users = unique_users[start:end]

        chunk_dedup = dedup[dedup['user_id'].isin(chunk_users)]

        merged = chunk_dedup.merge(item_artists, on='item_id', how='inner')

        del chunk_dedup
        gc.collect()

        exploded = merged.explode('artists')
        exploded = exploded[exploded['artists'].astype(str) != '']

        del merged
        gc.collect()

        user_artists = exploded.groupby('user_id')['artists'].nunique().reset_index().rename(columns={'artists': 'user_unique_artists'})

        del exploded
        gc.collect()

        results.append(user_artists)
        del user_artists
        gc.collect()

    final_df = pd.concat(results, ignore_index=True)
    
    del dedup, item_artists, results
    gc.collect()

    return final_df


def get_user_features(events, items):
    print("Агрегация общих признаков...")
    user_features = events.groupby('user_id').agg(
        user_total_listens=('item_id', 'count'),
        user_unique_tracks=('item_id', 'nunique'),
        user_first_listen=('started_at', 'min'),
        user_last_listen=('started_at', 'max')
    ).reset_index()

    last_global_date = events['started_at'].max()

    # Топ-3 жанры
    print("Выбор топ 3 жанров...")
    user_top_genres = get_user_top_genres_features(events, items)
    print("Присоединение топ 3 жанров...")
    user_features = user_features.merge(user_top_genres, on='user_id', how='left')

    del user_top_genres
    gc.collect()

    user_group = events.groupby('user_id')['item_id']

    week_ago = last_global_date - pd.Timedelta(days=7)
    month_ago = last_global_date - pd.Timedelta(days=30)

    # Количество прослушиваний за последние 7 дней
    print('Подсчет количества прослушиваний за последние 7 дней...')
    listens_7d = events[events['started_at'] >= week_ago].groupby('user_id').size().reset_index(name='user_listens_last_7d')
    # Количество прослушиваний за последние 30 дней
    print('Подсчет количества прослушиваний за последние 30 дней...')
    listens_30d = events[events['started_at'] >= month_ago].groupby('user_id').size().reset_index(name='user_listens_last_30d')

    print('Присоединение количества прослушиваний за последние 7 дней...')
    user_features = user_features.merge(listens_7d, on='user_id', how='left').fillna({'user_listens_last_7d': 0})
    print('Присоединение количества прослушиваний за последние 30 дней...')
    user_features = user_features.merge(listens_30d, on='user_id', how='left').fillna({'user_listens_last_30d': 0})

    del listens_7d, listens_30d
    gc.collect()

    # Возраст пользователя в системе
    print('Расчет возраста пользователя в системе...')
    user_features['user_lifetime_days'] = (
        last_global_date - user_features['user_first_listen']
    ).dt.days.astype(np.int32)

    # Количество дней с последнего прослушивания
    print('Подсчет количества дней с последнего прослушивания...')
    user_features['user_days_since_last_listen'] = (
        last_global_date - user_features['user_last_listen']
    ).dt.days.astype(np.int32)

    print('Подсчет уникальных  артистов пользователя...')
    user_artists = get_user_unique_artists(events, items)
    print("Присоединение уникальных артистов...")
    user_features = user_features.merge(user_artists, on='user_id', how='left').fillna({'user_unique_artists': 0})

    del user_artists
    gc.collect()

    print('Подсчет частоты повторов треков пользователя...')
    user_features['user_replay_rate'] = (
        user_features['user_total_listens'] / user_features['user_unique_tracks']
    ).astype(np.float32)

    print("Удаление лишних признаков...")
    user_features.drop(columns=['user_first_listen', 'user_last_listen'], inplace=True)
    
    gc.collect()
    return user_features


# In[17]:


user_features_train = get_user_features(events_train, items)

user_features_train.to_parquet("user_features_train.parquet")


# In[26]:


item_features_train = get_item_features(events_train)

item_features_train.to_parquet("item_features_train.parquet")


# # Ранжирование рекомендаций

# Построим ранжирующую модель, чтобы сделать рекомендации более точными. Отранжируем рекомендации.

# In[11]:


import polars as pl

als_pl = pl.scan_parquet("personal_als.parquet")
content_pl = pl.scan_parquet("content_recs.parquet")

als_pl = als_pl.select([
    pl.col("user_id"),
    pl.col("item_id"),
    pl.col("score").alias("als_score")
])


content_pl = content_pl.select([
    pl.col("user_id"),
    pl.col("item_id"),
    pl.col("score").alias("cnt_score")
])

candidates_lazy = als_pl.join(content_pl, on=['user_id', 'item_id'], how='outer')

candidates = candidates_lazy.collect()

candidates = candidates.with_columns([
    pl.coalesce(["user_id", "user_id_right"]).alias("user_id"),
    pl.coalesce(["item_id", "item_id_right"]).alias("item_id")
])

candidates = candidates.drop(["user_id_right", "item_id_right"]).to_pandas()


# In[3]:


events_train = pd.read_parquet("events_test.parquet")


# In[4]:


events_test = pd.read_parquet("events_test.parquet")


# In[5]:


split_date_for_labels = pd.to_datetime('2022-12-24')

split_date_idx = events_test['started_at'] < split_date_for_labels
events_labels = events_test[split_date_idx] 
events_test_2 = events_test[~split_date_idx]


# In[12]:


events_labels['target'] = 1

candidates_merged = candidates.merge(events_labels[['user_id', 'item_id', 'target']], on=['user_id', 'item_id'], how='left')

del candidates
gc.collect()

candidates_merged['target'] = candidates_merged['target'].fillna(0).astype(int)

candidates_merged['als_score'] = candidates_merged['als_score'].astype(np.float32)
candidates_merged['cnt_score'] = candidates_merged['cnt_score'].astype(np.float32)

candidates_to_sample = candidates_merged.groupby('user_id').filter(lambda x: x['target'].sum() > 0)

negative_samples = 4

candidates_to_train = pd.concat([
    candidates_to_sample.query("target == 1"),
    candidates_to_sample.query("target == 0")
    .groupby('user_id')
    .apply(lambda x: x.sample(n=negative_samples))
]).reset_index(drop=True)

del candidates_to_sample
gc.collect()


# In[13]:


candidates_to_train


# In[14]:


candidates_to_train.to_parquet('candidates_to_train.parquet')


# In[15]:


del candidates_to_train
gc.collect()


# In[18]:


import polars as pl

candidates_lf = pl.scan_parquet('candidates_to_train.parquet')
user_features_lf = pl.scan_parquet('user_features_train.parquet')
item_features_lf = pl.scan_parquet('item_features_train.parquet')

final_dataset_lf = (
    candidates_lf
    .join(user_features_lf, on='user_id', how='left')
    .join(item_features_lf, on='item_id', how='left')
    .fill_null(0)
)

final_dataset_lf.sink_parquet('final_train_dataset.parquet')


# In[19]:


final_train_dataset = pl.scan_parquet("final_train_dataset.parquet").collect().to_pandas()


# In[24]:


from catboost import CatBoostClassifier, Pool
import joblib

features = ['als_score', 'cnt_score'] + ['user_total_listens', 
    'user_unique_tracks', 'top_genre_1', 'top_genre_2', 'top_genre_3', 
    'genre_ratio', 'user_listens_last_7d', 'user_listens_last_30d', 
    'user_lifetime_days', 'user_days_since_last_listen', 
    'user_unique_artists', 'user_replay_rate'] + ['weekly_listenings', 
    'monthly_listenings', 'moving_avg_7d', 'velocity', 'absolute_growth', 'momentum']

target = 'target'

train_pool = Pool(
    data=final_train_dataset[features],
    label=final_train_dataset[target]
)

rank_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=100,
    loss_function='Logloss'
)

rank_model.fit(train_pool)

joblib.dump(rank_model, 'rank_model.pkl')


# In[7]:


events_inference = pd.concat([events_train, events_labels])

del events_train, events_labels
gc.collect()


# In[16]:


import numpy as np
from implicit.als import AlternatingLeastSquares

def load_als_model(filename):
    data = np.load(filename, allow_pickle=True)
    print(data['user_factors'])

    factors = data['user_factors'].shape[1]
    model = AlternatingLeastSquares(factors=factors)

    model.user_factors = data['user_factors']
    model.item_factors = data['item_factors']

    del data
    del factors

    return model


# In[18]:


als_model = load_als_model('als_model.npz')


# In[19]:


from implicit.nearest_neighbours import BM25Recommender

content_model = BM25Recommender()

content_model.load('content_model.npz')


# In[26]:


with open('user_item_matrix_train.pkl', 'rb') as f:
    user_item_matrix_train = joblib.load(f)


# In[ ]:


os.makedirs('candidates/inference', exist_ok=True)

recommend_with_batches(
    model=als_model,
    save_filename='candidates/inference/personal_als.parquet',
    user_item_matrix=user_item_matrix_train,
    user_encoder=user_encoder,
    item_encoder=item_encoder,
    batch_size=10000
)


# # Оценка качества

# Проверим оценку качества трёх типов рекомендаций: 
# 
# - топ популярных,
# - персональных, полученных при помощи ALS,
# - итоговых
#   
# по четырем метрикам: recall, precision, coverage, novelty.

# In[ ]:





# In[ ]:





# # === Выводы, метрики ===

# Основные выводы при работе над расчётом рекомендаций, рассчитанные метрики.

# In[ ]:





# In[ ]:




