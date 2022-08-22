from typing import List, Optional, Union, Dict
import numpy as np
import pandas as pd
import torch as t
from sentence_transformers import SentenceTransformer
import os
from datetime import date, datetime
from sentence_transformers.util import cos_sim
import uuid
from fastapi import FastAPI
import uvicorn

from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)


class Sentence2vector:
    """
    use model to convert sentence to vector
    """

    def __init__(self,
                 model_name: str = "hfl/chinese-roberta-wwm-ext",
                 device: str = 'cuda') -> None:

        logging.info("init sentence2vector")
        self.model = SentenceTransformer(
            model_name_or_path=model_name, device='cuda')
        self.model_name = model_name
        if t.cuda.is_available() and device == 'cuda':
            self.device = device
        else:
            self.device = 'cpu'

    def encode(self, text: Union[str, List[str]]):
        logging.info("sentence 2 vector encoding")
        if text is None:
            text = [""]

        if isinstance(text, str):
            text = [text]

        if len(text) > 1:
            show_progress_bar = True
        else:
            show_progress_bar = False
        text = [str(i) for i in text]
        vector = self.model.encode(
            text,
            device=self.device,
            show_progress_bar=show_progress_bar)

        # if t.cuda.is_available() and self.device == 'cuda':
        #     vector = self.model.encode(
        #         text,
        #         device=self.device,
        #         show_progress_bar=show_progress_bar)
        # else:
        #     vector = self.model.encode(text)

        return vector


class SVD:
    def __init__(self, file_dir: str = "QADIR"):

        self.s2v = Sentence2vector()
        logging.info("init search vector database")

        self.file_dir = file_dir

        os.makedirs(name=self.file_dir, exist_ok=True)

        self.title_file_path = file_dir + "/title_raw_data.csv"
        self.sim_file_path = file_dir + "/similar_raw_data.csv"

        if os.path.exists(self.title_file_path) and os.path.exists(self.sim_file_path):
            logging.info("load data from save file")

            # process title file
            title_df = pd.read_csv(self.title_file_path)
            title_df['rep_id'] = title_df['rep_id'].astype('str')
            if 'status_question' not in title_df.columns.tolist():
                title_df['status_question'] = 1
            if 'create_time' not in title_df.columns.tolist():
                title_df['create_time'] = datetime.now()

            if 'modify_time' not in title_df.columns.tolist():
                title_df['modify_time'] = datetime.now()

            # process similarity file
            sim_df = pd.read_csv(self.sim_file_path)
            sim_df['rep_id'] = sim_df['rep_id'].astype('str')
            if 'status_similar' not in sim_df.columns.tolist():
                sim_df['status_similar'] = 1

            if 'create_time' not in sim_df.columns.tolist():
                sim_df['create_time'] = datetime.now()

            if 'modify_time' not in sim_df.columns.tolist():
                sim_df['modify_time'] = datetime.now()

            sim_df['sim_index'] = np.arange(sim_df.shape[0]) +0#+ self.sim_df.shape[0]
            rep_id_4_list = title_df.loc[title_df['status_question'] == 0]['rep_id'].tolist(
            )
            sim_df.loc[sim_df['rep_id'].isin(rep_id_4_list), 'status_similar'] = 0
            self.title_df = title_df.copy()  # pd.concat([self.title_df, title_df.copy()])
            self.sim_df = sim_df.copy()  # pd.concat([self.sim_df, sim_df.copy()])

            # self.title_df = pd.concat([self.title_df, title_df.copy()])
            # self.sim_df = pd.concat([self.sim_df, sim_df.copy()])
        else:
            logging.info("load data from file to create a svd object")

            self.create_user_info()
            self.title_df = pd.read_csv(self.first_title_file)
            self.sim_df = pd.read_csv(self.first_simi_file)

        self.vector4sim = self.s2v.encode(self.sim_df['similarity'].tolist())
        self.vector4title = self.s2v.encode(self.title_df['title'].tolist())

    def create_user_info(self):
        """
        add info to this database
        """
        os.makedirs(name='init_user_dir', exist_ok=True)
        title_file = "init_user_dir/title_raw_data.csv"
        simi_file = "init_user_dir/similar_raw_data.csv"
        self.first_title_file = title_file
        self.first_simi_file = simi_file

        if not os.path.exists(title_file):
            uuid_str = uuid.uuid4().__str__()

            cui_ = pd.DataFrame({'title': ['你是谁'],
                                 'answer': ['由公众号【world of statistics】作者创建']})
            cui_['create_time'] = datetime.now()
            cui_['modify_time'] = datetime.now()
            cui_['rep_id'] = uuid_str
            cui_['status_question'] = 1

            # print(cui_)

            simi_cui = pd.DataFrame({
                'similarity': ['你叫什么名字', '你从哪里来的']
            })
            simi_cui['create_time'] = datetime.now()
            simi_cui['modify_time'] = datetime.now()
            simi_cui['rep_id'] = uuid_str
            simi_cui['status_similar'] = 1
            simi_cui['sim_index'] = np.arange(simi_cui.shape[0])
            # print(simi_cui)

            cui_.to_csv(title_file, index=False)
            simi_cui.to_csv(simi_file, index=False)

    def select_by_repid(self, temp_repid: str):
        """
        return title and similar df
        """
        logging.info("select a question by repid")

        temp_title_df = self.title_df.loc[self.title_df['rep_id']
                                          == temp_repid]
        temp_sim_df = self.sim_df.loc[(self.sim_df['rep_id'] == temp_repid) & (
                self.sim_df['status_similar'] == 1)]

        return temp_title_df, temp_sim_df

    def save_data2_file(self):
        self.title_df.to_csv(self.title_file_path, index=False)
        self.sim_df.to_csv(self.sim_file_path, index=False)

    def search_topn(self, search_text: str, topn: Optional[int]) -> pd.DataFrame:
        """
        return topN by query
        """
        logging.info("search topN by search text")
        if topn is None:
            topn = 5
        search_text_encoding = self.s2v.encode(search_text)

        # search with sim then generate scores and rep_id
        search_sim_score = cos_sim(
            search_text_encoding, self.vector4sim).numpy().flatten()
        search_sim_score_df = self.sim_df[['rep_id', 'status_similar']].copy().rename(
            columns={'status_similar': 'status'})
        search_sim_score_df['score'] = search_sim_score
        # search_sim_score_df['mask'] = self.sim_df['status_similar'] == 1
        search_sim_score_df = search_sim_score_df.loc[self.sim_df['status_similar'] == 1, :]

        search_title_score = cos_sim(
            search_text_encoding, self.vector4title
        ).numpy().flatten()
        search_title_score_df = self.title_df[['rep_id', 'status_question']].copy().rename(
            columns={'status_question': 'status'})
        search_title_score_df['score'] = search_title_score
        # search_title_score_df['mask'] = self.title_df['status_question'] == 1
        search_title_score_df = search_title_score_df.loc[self.title_df['status_question'] == 1, :]

        # concat
        total_value = pd.concat([search_sim_score_df, search_title_score_df])
        # total_value = total_value.loc[total_value['mask'], :]
        total_value['rank'] = total_value['score'].rank(ascending=False)
        total_value['re_rank'] = 1 / total_value['rank']
        finalrepid = total_value.query('''status ==1''').groupby(['rep_id']).agg(
            mrr=('re_rank', 'mean')
        ).sort_values(by='mrr', ascending=False).head(topn).reset_index(drop=False)

        return finalrepid

    def create_new(self, part_title: Dict, part_sim: Union[str, List[str]]):
        """
        create new item

        """
        logging.info("create a new item")
        if not isinstance(part_title, Dict):
            raise ValueError("part_title must be a dict")

        question_title = part_title.get('title', None)
        if question_title is None:
            raise ValueError("question title must be a str dont empty")

        rep_id = uuid.uuid4().__str__()

        clean_title_new_df = pd.DataFrame({'title': [question_title],
                                           'answer': [part_title.get('answer', None)],
                                           # 'ret_type': [rep_type],
                                           # 'store_id': [store_id],
                                           'rep_id': [rep_id],
                                           'status_question': [1],
                                           'create_time': [datetime.now()],
                                           'modify_time': [datetime.now()]
                                           })
        # print(clean_title_new_df.info())

        # ['rep_id', 'sim_title', 'status', 'create_time', 'modify_time', 'sim_index']
        if isinstance(part_sim, str):
            part_sim_list = part_sim.split('###')
        elif isinstance(part_sim, List):
            part_sim_list = part_sim
        clean_sim_new_df = pd.DataFrame({'similarity': part_sim_list})
        clean_sim_new_df['rep_id'] = rep_id
        clean_sim_new_df['status_similar'] = 1
        clean_sim_new_df['create_time'] = datetime.now()
        clean_sim_new_df['modify_time'] = datetime.now()
        start_sim_index = int(self.sim_df['sim_index'].max()) + 1
        # clean_sim_new_df['sim_index'] = np.arange(
        #     start_sim_index + 1, start_sim_index + 1 + len(part_sim_list))
        clean_sim_new_df['sim_index'] = np.arange(
            len(part_sim_list)) + self.sim_df.shape[0]

        self.sim_df = pd.concat(
            [self.sim_df, clean_sim_new_df]).reset_index(drop=True)
        self.title_df = pd.concat(
            [self.title_df, clean_title_new_df]).reset_index(drop=True)

        # update vector
        vector4sim_new = self.s2v.encode(part_sim_list)
        vector4title_new = self.s2v.encode(question_title)
        self.vector4sim = np.vstack([self.vector4sim, vector4sim_new])
        self.vector4title = np.vstack([self.vector4title, vector4title_new])

    def delete_title(self, delete_title: Optional[Union[str, List[str]]] = None):
        """
        delete item
        1. delete_title: delete a title ,need a rep_id

        """

        logging.info("delete a title or delete a sim_title")
        if delete_title is not None:
            # if isinstance(delete_title, List):
            #     delete_title = delete_title
            if isinstance(delete_title, str):
                if delete_title.find(',') != -1:
                    delete_title = delete_title.split(',')
                delete_title = [delete_title]

            self.title_df.loc[self.title_df['rep_id'].isin(
                delete_title), 'status_question'] = 0
            self.sim_df.loc[self.sim_df['rep_id'].isin(
                delete_title), 'status_similar'] = 0

    def delete_similar(self, sim_index: Optional[Union[int, List[int]]] = None):
        """
        2. delete_sim_title: 删除一个相似问法，需要传递的是相似问法的文本
        """
        logging.info("delete similar title")
        if sim_index is not None:
            if isinstance(sim_index, int):
                sim_index = [sim_index]
            if isinstance(sim_index, List):
                # delete_sim_title = delete_sim_title.split(',')
                print(f'simi_index: {sim_index}')

                self.sim_df.loc[self.sim_df['sim_index'].isin(sim_index), 'status_similar'] = 0

    def update_answer(self, update_repid: Optional[str] = None, new_answer: Optional[str] = None):
        """
        update answer
        """
        logging.info("update answer by repid")
        if update_repid is not None and new_answer is not None:
            self.title_df.loc[self.title_df['rep_id']
                              == update_repid, 'answer'] = new_answer

    def add_similar(self, rep_id: str = None, part_sim_list: Union[str, List[str]] = None):
        """
        in a spec rep_id , add new similarity title
        """
        logging.info("add sim title to database")
        if rep_id is not None:
            if part_sim_list is not None:
                if isinstance(part_sim_list, str):
                    part_sim_list = part_sim_list.split('###')
                elif isinstance(part_sim_list, List):
                    part_sim_list = part_sim_list
                clean_sim_new_df = pd.DataFrame({'similarity': part_sim_list})
                clean_sim_new_df['rep_id'] = rep_id
                clean_sim_new_df['status_similar'] = 1
                clean_sim_new_df['create_time'] = datetime.now()
                clean_sim_new_df['modify_time'] = datetime.now()
                # start_sim_index = int(self.sim_df['sim_index'].max()) + 1
                # clean_sim_new_df['sim_index'] = np.arange(
                #     start_sim_index + 1, start_sim_index + 1 + len(part_sim_list))
                clean_sim_new_df['sim_index'] = np.arange(
                    len(part_sim_list)) + self.sim_df.shape[0]
                self.sim_df = pd.concat(
                    [self.sim_df, clean_sim_new_df]).reset_index(drop=True)

                vector4sim_new = self.s2v.encode(part_sim_list)
                self.vector4sim = np.vstack([self.vector4sim, vector4sim_new])


svd = SVD()
# svd.init_by_file()

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/search_topn")
def searchTopN(search_text=None, topn=2):
    topn = int(topn)
    data = svd.search_topn(search_text=search_text, topn=topn)
    data = data.to_json(orient='records')
    return data


@app.get("/select_by_repid")
def select_by_repid(rep_id):
    temp_title_df, temp_sim_df = svd.select_by_repid(rep_id)
    data = {'title': temp_title_df.to_json(orient='records'),
            'sim': temp_sim_df.to_json(orient='records')}

    return data


@app.get("/create_new")
def create_new(part_title=None, part_answer=None, part_sim=None):
    part_title = {
        'title': part_title,
        'answer': part_answer
    }
    part_sim = part_sim.split('###')
    svd.create_new(part_title, part_sim)


@app.get("/add_similar")
def add_similar(rep_id: str = None, simi_list: str = None):
    svd.add_similar(rep_id=rep_id, part_sim_list=simi_list)


@app.get("/delete_title")
def delete_title(delete_title=None):
    svd.delete_title(delete_title)


@app.get("/delete_similar")
def delete_similar(delete_similar=None):
    if delete_similar.find('###') != -1:
        delete_similar = [int(i) for i in delete_similar.split('###')]
    else:
        delete_similar = int(delete_similar)
    svd.delete_similar(sim_index=delete_similar)


@app.get("/update_answer")
def update_answer(update_repid=None, new_answer=None):
    svd.update_answer(update_repid, new_answer)


@app.get("/saved2f")
def save_data2file():
    svd.save_data2_file()


if __name__ == '__main__':
    uvicorn.run(app='backend_new:app', host="0.0.0.0",
                port=8010, reload=False, debug=False)
