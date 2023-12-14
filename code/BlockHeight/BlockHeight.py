from operator import truediv
import basictype as bt
import json
import collections
import sys
import time
import logging
import random
import configparser
import os

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import pandas as pd
from pandas import Series,DataFrame
mpl.use('Agg')

class BH_blockchain:

    tx_count = 0
    begin_index = 0
    end_index = 0

    timestamp = 0

    node_number  = 0

    nodes = []

    send_rate = 0

    tx_list = collections.deque([])

    tx_send = []
    tx_finish = {}
    tx_exceed = {}
    tx_pop = {}

    txpool_len = []
    avg_txpool_len = []

    block_interval = []
    avg_block_interval = []

    txcount_in_block = []
    failed_tx_in_block = 0
    valid_tx_in_block = 0

    txs_exectime_list = []
    avg_tx_delay = []
    avg_remain_time = []

    resource_ratio =[]
    tx_success_ratio = []
    tps_per_block = []

    txpool_pop_exceedtx_num = 0

    txvalidtime_list = []
    blockinterval_list = []

    estimate_block_interval = 0
    txLimit = 0

    validbutpop = 0
    invalidpop = 0
    
    packtime = []

    def __init__(self,config,data_path,send_rate) -> None:
        logging.info("*********** BH_blockchain start ***********")
        self.config = config
        self.data_path = data_path
        self.begin_index = config.getint("init","begin_index")
        self.end_index = config.getint("init","end_index")
        self.node_number = config.getint("init","node_number")
        self.send_rate = send_rate
        self.bc = bt.BlockChain()
        self.initNodes()

        self.loadBlockinterval(config.getint("BH","block_interval_para"))
        self.loadTxDeadline(config.getint("BH","low"),config.getint("BH","high"))

        self.estimate_block_interval = config.getint("BH","estimate")
        self.txLimit = config.getint("BH","Blocktxlimit")
        self.loadTX(self.begin_index,self.end_index)
        self.initSendTx(config.getint("init","initTxCount"))
        
        logging.info("transaction send rate :{}".format(self.send_rate))

    def initNodes(self):
        gas_file = "../node/gas_distribution.json"
        with open(gas_file,'r') as fin:
            data = json.load(fin)
        gas_list = data["nodes_gas"]

        time_drift_file = "../node/time_drift_distribution.json"
        with open(time_drift_file,'r') as fin:
            data = json.load(fin)
        time_drift_list = data["nodes_time_drift"]

        for i in range(self.node_number):
            self.nodes.append(
                bt.node(i,int(gas_list[i]),time_drift_list[i])
            )
    
    def loadBlockinterval(self,lambda_para):
        file_name = "../block_interval/poisson_interval_{}.json".format(lambda_para)

        with open(file_name,'r') as fin:
            data = json.load(fin)

        self.blockinterval_list = data["poisson_random_numbers"]

    def loadTxDeadline(self,low, high):
        file_name = "../tx_deadline/poisson_deadline_{}_{}.json".format(low, high)

        with open(file_name,'r') as fin:
            data = json.load(fin)

        self.txvalidtime_list = data["poisson_random_numbers"]
        print(len(self.txvalidtime_list))

    def loadTX(self,begin_index,end_index):

        with open("../tx_data/tx_{}_{}.json".format(begin_index,end_index),'r') as ff:
            data = json.load(ff)
        i = 0

        for transaction in data:

            tx = bt.Transaction(
                transaction["tx_hash"],
                transaction["gas_used"],
                transaction["size"],
                self.txvalidtime_list[i],
                transaction["exec_period"]
                )
            
            i += 1 
            self.tx_list.append(tx)

        self.tx_list_len = len(self.tx_list)
        self.tx_count = self.tx_list_len
        logging.info("total tx num:{}".format(self.tx_list_len))

    def initSendTx(self,txnum):
        # The initial transaction in the transaction pool
        for i in range(txnum):
            tx = self.tx_list.popleft()
            tx.start_time = 0
            tx.ddl = tx.start_time + tx.valid_period
            tx.block_limit = int(tx.ddl / self.estimate_block_interval)
            self.bc.tx_pool.append(tx)
        self.txpool_len.append(txnum)
        self.avg_txpool_len.append(txnum)

    def is_finished(self):
        if len(self.tx_finish.keys()) + len(self.tx_exceed.keys()) == self.tx_list_len:
            return True
        return False

    def leaderSelect(self):
        self.leader = random.randint(0,self.node_number - 1)
        logging.info("[leader select] leader:{}".format(self.leader))

    def sortTxpool(self):
        time0 = time.time()
        self.bc.tx_pool = collections.deque(
            
            sorted(self.bc.tx_pool,key=lambda x: x.block_limit)
        )
        time1 = time.time()
        return self.bc.tx_pool,time1-time0

    def leaderCreateProposal(self):
        tx_pool,sort_time = self.sortTxpool()

        logging.info("txpool len:{}".format(len(self.bc.tx_pool)))

        height = len(self.bc.blocks)
        gas_total = self.nodes[self.leader].gas_total

        txs = []
        # poped transactions
        self.tx_pop[height] = []

        while gas_total > 0 and len(tx_pool) > 0:
            
            tx = tx_pool[0]

            if tx.block_limit < height:
                tx_pool.popleft()
                self.txpool_pop_exceedtx_num += 1
                tx.pop_bh = height
                self.tx_pop[height].append(tx)
                self.tx_exceed[tx.tx_hash] = tx
            else:
                tx_gas = tx.gas_used
                if tx_gas < gas_total:
                    tx = tx_pool.popleft()
                    tx.pack_bh = height
                    txs.append(tx)
                    gas_total -= tx.gas_used
                    # logging.info("proposal tx ddl:{}| tx-bh {}".format(tx.ddl,tx.block_limit))
                else:
                    break
        return txs
    
    def consensus(self,txs):
        nodes_time = []
        for node in self.nodes:
            node.time += self.blockinterval_list[len(self.bc.blocks)-1]
            nodes_time.append(node.time)
        logging.info("nodes timestamp:{}".format(nodes_time)) 

        block = bt.Block(
            self.leader,
            len(self.bc.blocks),
            self.nodes[self.leader].time,
            txs
        )
        self.bc.blocks.append(block)
        
    def update(self):

        # update average txpool length
        self.txpool_len.append(len(self.bc.tx_pool))
        self.avg_txpool_len.append(np.mean(self.txpool_len))

        if len(self.bc.blocks) != 1:
            interval = self.bc.blocks[-1].timestamp - self.bc.blocks[-2].timestamp
            self.block_interval.append(interval)

        # update pack tx state
        used_resource = 0
        txs = self.bc.blocks[-1].txs
        logging.info("txs count:{}".format(len(txs)))
        # logging.info("real time:{}".format(self.bc.blocks[-1].timestamp))
        exceed_tx_in_block = 0
        valid_tx_in_block = 0
        for tx in txs:
            vote  = 0
            tx_endtime_list = []
            for node in self.nodes:
                if node.time <= tx.ddl:
                    vote += 1
                    tx_endtime_list.append(node.time)
            if vote > len(self.nodes) * 2 / 3:
                tx.state = "pack_valid"
                tx.end_time = np.mean(tx_endtime_list)
                self.tx_finish[tx.tx_hash] = tx
                valid_tx_in_block += 1
                used_resource += tx.gas_used
            else:
                tx.state = "pack_invalid"
                self.tx_exceed[tx.tx_hash] = tx
                exceed_tx_in_block += 1

        logging.info("valid_tx_in_block:{}".format(valid_tx_in_block))
        logging.info("invalid_tx_in_block:{}".format(exceed_tx_in_block))
        # update valid txcount in a single block
        self.valid_tx_in_block += valid_tx_in_block 
        self.failed_tx_in_block += exceed_tx_in_block

        # update pop tx state
        for tx in self.tx_pop[self.bc.blocks[-1].block_number]:
            vote = 0
            for node in self.nodes:
                if node.time <= tx.ddl:
                    vote += 1
            if vote > len(self.nodes) * 2 / 3:
                self.validbutpop += 1
                tx.state = "pop_valid"
            else:
                self.invalidpop += 1
                tx.state = "pop_invalid"
        
        # update tx success ratio
        tx_success_ratio = len(self.tx_finish.keys()) / (len(self.tx_finish.keys()) + len(self.tx_exceed.keys()))
        self.tx_success_ratio.append(tx_success_ratio)

        # update avg TPS
        self.tps_per_block.append(len(self.tx_finish.keys()) / self.bc.blocks[-1].timestamp)

        # update resource ratio
        self.resource_ratio.append(used_resource / self.nodes[self.leader].gas_total)

    def sendTx(self):
        # logging.info("======send_tx========")
        if len(self.bc.blocks) > 1:
            last_timestamp = self.bc.blocks[-2].timestamp
        else:
            last_timestamp = 0

        # send transactions
        start_timestamp = math.ceil(last_timestamp) 
        end_timestamp = int(self.bc.blocks[-1].timestamp)
        
        has_tx = True
        while has_tx and start_timestamp < end_timestamp:
            logging.info("sendrate:{}".format(self.send_rate))
            for i in range(self.send_rate):
                try:
                    tx = self.tx_list.popleft()
                except Exception as ex:
                    has_tx = False
                    break
                tx.start_time = start_timestamp
                tx.ddl = tx.start_time + tx.valid_period
                
                # calculate the latest packed block height for each transaction
                if int(tx.ddl / self.estimate_block_interval) == 0:
                    tx.block_limit = 0
                else:
                    tx.block_limit = int((tx.ddl - last_timestamp) / self.estimate_block_interval) + (len(self.bc.blocks) - 1) - 1

                self.bc.tx_pool.append(tx)
                self.tx_send.append(tx)
            start_timestamp += 1

    def logOut(self):
        logging.info("[Ep.{}] time: {:.6}  avgTPS: {:.6f} | success txcount: {} | exceed txcount: {} | tx_ratio: {:.6}".format(
            len(self.bc.blocks)-1,
            self.bc.blocks[-1].timestamp,
            self.tps_per_block[-1],
            len(self.tx_finish.keys()),
            len(self.tx_exceed.keys()),
            self.tx_success_ratio[-1]
        ))

    def txdl_handle_states(self):
        state = []
        valid_period = []
        for key in self.tx_finish.keys():
            state.append(self.tx_finish[key].state)
            valid_period.append(self.tx_finish[key].valid_period)
        for key in self.tx_exceed.keys():
            state.append(self.tx_exceed[key].state)
            valid_period.append(self.tx_exceed[key].valid_period)
        
        df = DataFrame({"state":state,"deadline":valid_period})

        sns.histplot(data = df,x="deadline",hue="state",hue_order=["pack_valid","pack_invalid","pop_valid","pop_invalid"], multiple="stack")
        plt.savefig(self.data_path+"handle_state_Distribution_{}_{}.jpg".format(config.getint("BH","low"),config.getint("BH","high")))

    def statistics(self):
        logging.info("======statistics========")
        logging.info("avg validtx in single block:{}".format(self.valid_tx_in_block / len(self.bc.blocks)))
        logging.info("avg failedtx in single block:{}".format(self.failed_tx_in_block / len(self.bc.blocks)))
        logging.info("total valid tx in blocks:{} | ratio:{}".format(self.valid_tx_in_block,self.valid_tx_in_block/self.tx_count))
        logging.info("total invalid tx in blocks:{}| ratio:{}".format(self.failed_tx_in_block,self.failed_tx_in_block/self.tx_count))
        logging.info("txpool pop exceed blockheight tx num:{} | ratio:{}".format(self.txpool_pop_exceedtx_num,self.txpool_pop_exceedtx_num/self.tx_count))
        logging.info("txpool pop num (precise time valid tx):{} | ratio:{}".format(self.validbutpop,self.validbutpop/self.tx_count))
        logging.info("txpool pop num (precise time invalid tx):{} | ratio:{}".format(self.invalidpop,self.invalidpop/self.tx_count))
        
        logging.info("avg block interval:{}".format(np.mean(self.block_interval)))
        logging.info("resource ratio:{}".format(np.mean(self.resource_ratio)))
        logging.info([i.timestamp for i in self.bc.blocks])
        logging.info(self.tx_success_ratio)
        logging.info(self.tps_per_block)
        logging.info("avg pack time:{}".format(np.mean(self.packtime)))
        self.txdl_handle_states()

    def simulation(self):
        
        while not self.is_finished():
            logging.info("[Block {} start]".format(len(self.bc.blocks)))
            t0 = time.time()
            self.leaderSelect()
            txs = self.leaderCreateProposal()
            t1 = time.time()
            t = t1-t0
            self.packtime.append(t)

            self.consensus(txs)
            self.update()
            self.sendTx()
            self.logOut()
        
        self.statistics()
        
def mkdirlg(data_path): 
    if not os.path.exists(data_path):
        os.makedirs(data_path) 

if __name__ == "__main__":

    send_rate = int(sys.argv[1])

    # logging config
    
    config = configparser.ConfigParser()
    config.read('./config.ini')
    data_path= "./result/blockinterval_{}/deadline_{}_{}/tx_sendrate_{}/".format(
            config.getint("BH","block_interval_para"),config.getint("BH","low"),config.getint("BH","high"),send_rate)
    mkdirlg(data_path)

    logging.basicConfig(
        filename = data_path+"result.log",
        level = logging.INFO,                 
        format = "%(asctime)s %(levelname)s %(message)s ",
        datefmt = '[%d] %H:%M:%S'
    )
    bc = BH_blockchain(config,data_path,send_rate)

    bc.simulation()