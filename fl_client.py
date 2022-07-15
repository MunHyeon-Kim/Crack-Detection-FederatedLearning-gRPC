from __future__ import print_function

import os
import time
import random
import pickle
import grpc
from transport_pb2 import Scalar, transportRequest, ReadyReq, UpdateReq, VersionReq, State
from transport_pb2_grpc import TransportServiceStub

from client_fit_model import learning_fit


def request_parameter():
    update_request = [UpdateReq(type="P")]
    for ur in update_request:
        yield transportRequest(update_req=ur)


def request_ready():
    global client_name

    configuration = dict()
    configuration['current_round'] = Scalar(scint32=0)

    client_name = f'client{random.randint(1, 100000)}'
    ready_request = [ReadyReq(type='R', cname=client_name, state=State.ON, config=configuration)]
    for rr in ready_request:
        yield transportRequest(ready_req=rr)


# return transportRequest(ready_req=ReadyReq(type='READY', cname='client1', state=State.ON, config=configuration))

## send logs file about tensorboard
def get_file_chunks(filename):
    CHUNK_SIZE = 100 * 1024 * 1024
    with open(filename, 'rb') as f:
        while True:
            piece = f.read(CHUNK_SIZE)
            if len(piece) == 0:
                return
            yield transportRequest(
                update_req=UpdateReq(type="L", buffer_chunk=piece, title=filename, file_len=len(piece)))


def send_logs(stub, in_file_name):
    chunks_generator = get_file_chunks(in_file_name)
    logs_response = stub.transport(chunks_generator)
    for lr in logs_response:
        print(f"Finish deliver file: {lr.update_rep.title}, type: {lr.update_rep.type}")


##
## send training state to server
def request_training(nclient):
    training_request = [UpdateReq(type="T", cname=nclient, state=State.TRAINING)]
    for tr in training_request:
        yield transportRequest(update_req=tr)


def request_traindone(nclient, cr, bc):
    traindone_request = [
        UpdateReq(type="D", buffer_chunk=pickle.dumps(bc), state=State.TRAIN_DONE, cname=nclient, current_round=cr)]
    for tr in traindone_request:
        yield transportRequest(update_req=tr)


def request_model_version(mv, cr):
    configuration = dict()
    configuration['model_version'] = Scalar(scint32=mv)
    configuration['current_round'] = Scalar(scint32=cr)
    version_request = [VersionReq(type="P", config=configuration)]
    for vr in version_request:
        yield transportRequest(version_req=vr)


def send_message(stub):
    global client_name

    ready_state = False
    # ready client
    print("### Ready Client ###")
    ready_info_dict = dict()
    ready = request_ready()
    response_ready = stub.transport(ready)
    for rs in response_ready:
        if rs.ready_rep.config['state'].scstring == "SW":
            ready_state = True
            ready_info_dict['cr'] = rs.ready_rep.config['current_round'].scint32
            ready_info_dict['mtr'] = rs.ready_rep.config['max_train_round'].scint32
            # ready_info_dict['ga'] = rs.ready_rep.config['model_acc'].scfloat
            # ready_info_dict['gl'] = rs.ready_rep.config['model_loss'].scfloat
            ready_info_dict['tmt'] = rs.ready_rep.config['model_type'].scstring
            ready_info_dict['mv'] = rs.ready_rep.config['model_version'].scint32

    if ready_state:
        # update client
        print("### Request Global Model Parameter ###")
        update = request_parameter()
        response_update = stub.transport(update)
        for ru in response_update:
            class_for_learning = learning_fit(ready_info_dict['tmt'], 1, 128, ru.update_rep.buffer_chunk,
                                              ready_info_dict['cr'])

        # train client
        training = request_training(client_name)
        response_training = stub.transport(training)

        class_for_learning.manage_train(cr=ready_info_dict['cr'])  # model fit
        '''
		# update complete
		## send logs file to server
		print("### Upload model training log files ###")
		for root, dirs, files in os.walk('./send_logs/logs'):
			for fname in files:
				full_fname = os.path.join(root, fname)
				send_logs(stub, full_fname)
		'''

        get_params = list()
        while ready_info_dict['cr'] <= ready_info_dict['mtr']:
            # train done
            print("### Deliver model state: TRAIN DONE to server ###")
            with open('./saved_weight/weights.pickle', 'rb') as fr:
                get_params = pickle.load(fr)
            traindone = request_traindone(client_name, ready_info_dict['cr'], get_params)
            response_traindone = stub.transport(traindone)  ## 나 학습 다했어!

            oneres_traindone = None;
            oneres_newround = None
            for rt in response_traindone:
                oneres_traindone = rt

            print(f"### Received from state {oneres_traindone.update_rep.config['state'].scstring} ###")
            # case 1: still learning other model -> state: RESP_ACY
            if oneres_traindone.update_rep.config['state'].scstring == 'RESP_ACY':  ## 아직 다른거 학습중이야 기다려
                change_model_version = False
                while True:  ## 응...
                    if change_model_version:
                        break
                    time.sleep(20)  ## 30초만 자야지

                    # check model version
                    version = request_model_version(ready_info_dict['mv'], ready_info_dict['cr'])  ## 끝났어?
                    response_version = stub.transport(version)
                    for rv in response_version:
                        oneres_newround = rv
                    if oneres_newround.version_rep.state == State.NOT_WAIT:  ## 어 끝났어!
                        ready_info_dict['cr'] = oneres_newround.version_rep.config['current_round'].scint32
                        ready_info_dict['mv'] = oneres_newround.version_rep.config['model_version'].scint32
                        change_model_version = True
                    ## 아니 안끝났어 더 기다려!
                # train next round
                get_params = class_for_learning.manage_train(params=oneres_newround.version_rep.buffer_chunk,
                                                             cr=ready_info_dict['cr'])  ## 다음 라운드 학습해야지
            # case 2: finish learning one round -> state: RESP_ARY
            elif oneres_traindone.update_rep.config['state'].scstring == 'RESP_ARY':  ## 바로 다음 라운드 학습해~
                # train client
                training = request_training(client_name)
                response_training = stub.transport(training)  ## 나 학습 시작한다~

                ready_info_dict['cr'] = oneres_traindone.update_rep.config['current_round'].scint32
                ready_info_dict['mv'] = oneres_traindone.update_rep.config['model_version'].scint32

                get_params = class_for_learning.manage_train(params=oneres_traindone.update_rep.buffer_chunk,
                                                             cr=ready_info_dict['cr'])  ## 다음 라운드 학습!

            ## for root...		## logs파일 보내는 코드 함수화해서 여기에 넣기
            # case 3: finish all round training
            elif oneres_traindone.update_rep.config['state'].scstring == 'FIN':  ## 학습 끝났어!
                ready_info_dict['cr'] = oneres_traindone.update_rep.config['current_round'].scint32
                ready_info_dict['mv'] = oneres_traindone.update_rep.config['model_version'].scint32

                print("all training finish")
            # ??? 여기부터 구현해야함


def run():
    options = [('grpc.max_receive_message_length', 512 * 1024 * 1024),
               ('grcp.max_send_message_length', 512 * 1024 * 1024)]
    channel = grpc.insecure_channel('localhost:8889', options=options)
    stub = TransportServiceStub(channel)
    send_message(stub)


if __name__ == '__main__':
    client_name = ""
    run()
