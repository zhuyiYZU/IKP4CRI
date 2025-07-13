# -*- coding: utf-8 -*-
import logging
import subprocess
import time
from itertools import product
import random
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)  # 配置日志记录器

    l = ['DuDialRec']#
    # l = ['inspired']
    batch_sizes = {16}
    # batch_sizes = {i for i in (16,32,64)}
    # learning_rates = {i for i in ('3e-5','4e-5','5e-5')}
    learning_rates = {'5e-5'}
    # kptw_lr = {0.02,0.03,0.04,0.05}
    # shots = {i for i in (20,30,40,50)}
    shots = {40}
    seeds = {i for i in range(90,100)}
    # seeds = random.randint(0,1000)
    # seeds = {80}
    template_id = {0}
    verbalizer = {'manual'}
    for n, t, j, i, k, v ,m in product(l, template_id, seeds, batch_sizes, learning_rates , verbalizer,shots):
        cmd = (
            f"python fewshot.py --result_file ./result2/dudialrec/others.txt "
            # f"--label_result ./i_label/shot{m}bs{i}seed{j}lr{k}.txt "
            f"--dataset {n} --template_id {t} --seed {j} "
            f"--batch_size {i} --shot {m} "
            f"--learning_rate {k} --verbalizer {v}"
        )

        logging.info(f"Executing command: {cmd}")
        print(cmd)
        try:
            subprocess.run(cmd, shell=True, check=True)
            logging.info(f"Command executed successfully: {cmd}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed: {cmd}. Error: {e.stderr.decode().strip()}")

        time.sleep(1)
